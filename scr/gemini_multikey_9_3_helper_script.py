#!/usr/bin/env python3
"""
gemini_multikey.py — ENHANCED VERSION
High-speed multi-key rotation with parallel processing capabilities
"""

import os
import json
import time
import logging
import concurrent.futures
import threading
from logging.handlers import RotatingFileHandler
import sqlite3
import argparse
import pathlib
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from pathlib import Path

# ========================
# Constants
# ========================
# Timeout constants
PER_REQUEST_TIMEOUT = 90      # hard cap for each API call
FUTURE_RESULT_TIMEOUT = 120   # how long to wait for each future.result()
OVERALL_BATCH_TIMEOUT = 120 * 5  # max total time to wait for as_completed (optional)
MAX_WORKERS_DEFAULT = 1       # REDUCED from 8 to 1 to prevent 429 errors

# ========================
# Directory Structure
# ========================
BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

# Create directories if they don't exist
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ========================
# Additional Constants
# ========================
MAX_LOG_SIZE = 1_000_000
LOG_FILE = LOG_DIR / "gemini_multikey.log"
SQLITE_DB = DATA_DIR / "gemini_interactions.db"
JSON_LOG = DATA_DIR / "gemini_interactions_log.json"
CONFIG_FILE = BASE_DIR / "gemini_config.json"
KEY_VALIDATION_CACHE = DATA_DIR / ".key_validation_cache.json"
KEY_USAGE_FILE = DATA_DIR / ".key_usage.json"

# Rate limiting constants
DAILY_QUOTA_PER_KEY = 20        # free calls per key per day
RATE_PER_MIN = 15              # safety RPM (requests per minute)
BURST = 5                      # burst capacity (token bucket size)
COOLDOWN_ON_429 = 60            # seconds to cooldown on 429

# Available Gemini models
AVAILABLE_MODELS = {
    "flash-lite": "gemini-2.5-flash-lite",
    "flash": "gemini-2.5-flash", 
    "pro": "gemini-2.5-pro"
}
DEFAULT_MODEL = "gemini-2.5-flash"

# ========================
# Enhanced Logging Setup
# ========================
def setup_logging(verbose: bool = False):
    """Setup logging with configurable levels"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # FIX: Corrected date format from "%Y-%m-d" to "%Y-%m-%d"
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", 
        "%Y-%m-%d %H:%M:%S"
    )

    logger = logging.getLogger("gemini_multikey")
    # Prevent adding handlers multiple times if setup_logging is called repeatedly
    if logger.handlers:
        # update level and return existing logger
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    file_handler = RotatingFileHandler(
        LOG_FILE, 
        maxBytes=MAX_LOG_SIZE, 
        backupCount=3
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False  # avoid double logging

    return logger

# Initialize logger
logger = setup_logging()

# ========================
# SDK Imports
# ========================
try:
    import google.generativeai as genai
    HAS_GENAI = True
    logger.debug("Using google.generativeai SDK.")
except Exception as e:
    HAS_GENAI = False
    logger.warning(f"GenAI SDK not installed: {e}")

import requests
from PyPDF2 import PdfReader

# ========================
# Enhanced Config Loader
# ========================
def load_config(config_path: Path = CONFIG_FILE) -> Dict[str, Any]:
    """Load config.json with enhanced validation and graceful failure handling."""
    default_config = {
        "google_api_keys": [],
        "api_key_labels": [],
        "default_model": DEFAULT_MODEL
    }
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return default_config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        logger.error(f"Failed to parse config file: {e}")
        return default_config

    # Normalize API keys to list
    if isinstance(cfg.get("google_api_keys"), str):
        cfg["google_api_keys"] = [cfg["google_api_keys"]]
        logger.warning("Single Google API key detected. Converted to list.")
    
    # Ensure we have lists
    cfg["google_api_keys"] = cfg.get("google_api_keys", [])
    cfg["api_key_labels"] = cfg.get("api_key_labels", [])
    
    # Validate we have some keys
    if not cfg["google_api_keys"]:
        logger.warning("No API keys found in configuration")
    
    logger.info(f"Loaded configuration with {len(cfg['google_api_keys'])} API keys")
    return cfg

# ========================
# High-Speed Multi-Key Engine
# ========================
class HighSpeedMultiKeyEngine:
    """High-speed parallel processing engine with flexible key management"""
    
    def __init__(self, api_keys: List[str], labels: Optional[List[str]] = None, 
                 model: str = DEFAULT_MODEL, max_workers: int = None,
                 validate_keys: bool = False):
        if not HAS_GENAI:
            raise RuntimeError("GenAI SDK not found. Install 'google-generativeai'.")
        
        self.api_keys = api_keys or []
        if not self.api_keys:
            raise RuntimeError("No API keys provided.")
        
        self.labels = labels or [f"key{i}" for i in range(len(self.api_keys))]
        self.model = model
        # FIX: Use more reasonable max_workers calculation
        self.max_workers = max_workers or max(1, min(len(self.api_keys) * 2, 16))
        
        # Thread-safe statistics
        self.stats_lock = threading.RLock()
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "key_usage": {label: 0 for label in self.labels},
            "start_time": None,
            "total_processing_time": 0
        }
        
        # Key rotation state
        self.key_cycle = cycle(zip(self.api_keys, self.labels))
        self.key_lock = threading.Lock()
        
        # Key validation cache
        self._validation_cache_path = KEY_VALIDATION_CACHE
        self._validation_cache_duration = timedelta(days=1)
        
        # Per-key cooldown tracking (deprecated, now using key_usage_state)
        self.key_cooldowns = {label: 0 for label in self.labels}
        
        # Initialize key usage state with persistent storage
        self.key_usage_state = self._load_key_usage()
        today = datetime.now().strftime("%Y-%m-%d")
        for label in self.labels:
            if label not in self.key_usage_state:
                self.key_usage_state[label] = {
                    "date": today,
                    "count": 0,
                    "tokens": BURST,
                    "last_refill": time.time(),
                    "cooldown_until": 0
                }
            else:
                # Reset daily counts if it's a new day
                if self.key_usage_state[label]["date"] != today:
                    self.key_usage_state[label] = {
                        "date": today,
                        "count": 0,
                        "tokens": BURST,
                        "last_refill": time.time(),
                        "cooldown_until": 0
                    }
        
        logger.info(
            f"Initialized {self.__class__.__name__} with {len(self.api_keys)} keys | "
            f"Model: {self.model} | Max Workers: {self.max_workers}"
        )
        
        # Auto-validate keys once a day unless explicitly disabled
        if validate_keys or self._should_validate_keys():
            self.validate_all_keys()

    def _load_key_usage(self):
        """Load key usage state from persistent storage"""
        if not KEY_USAGE_FILE.exists():
            return {}
        try:
            data = json.loads(KEY_USAGE_FILE.read_text())
            # Ensure all entries have the required fields
            for key, value in data.items():
                if "date" not in value:
                    value["date"] = datetime.now().strftime("%Y-%m-%d")
                if "count" not in value:
                    value["count"] = 0
                if "tokens" not in value:
                    value["tokens"] = BURST
                if "last_refill" not in value:
                    value["last_refill"] = time.time()
                if "cooldown_until" not in value:
                    value["cooldown_until"] = 0
            return data
        except Exception as e:
            logger.error(f"Failed to load key usage state: {e}")
            return {}

    def _save_key_usage(self):
        """Save key usage state to persistent storage"""
        try:
            KEY_USAGE_FILE.write_text(json.dumps(self.key_usage_state, indent=2))
            logger.debug(f"Saved key usage state to {KEY_USAGE_FILE}")
        except Exception as e:
            logger.error(f"Failed to save key usage state: {e}")

    def _refill_tokens(self, label):
        """Refill tokens based on time elapsed since last refill"""
        if label not in self.key_usage_state:
            return
        
        st = self.key_usage_state[label]
        now = time.time()
        elapsed = now - st["last_refill"]
        # Refill at rate of RATE_PER_MIN tokens per minute
        st["tokens"] = min(BURST, st["tokens"] + elapsed * (RATE_PER_MIN / 60))
        st["last_refill"] = now

    def _can_use_key(self, label):
        """Check if a key can be used based on all rate limiting rules"""
        if label not in self.key_usage_state:
            return False
        
        st = self.key_usage_state[label]
        
        # Rule 1: Check if key is in cooldown (429)
        if time.time() < st["cooldown_until"]:
            logger.debug(f"Key {label} is in cooldown until {st['cooldown_until']}")
            return False
        
        # Rule 2: Check daily quota
        if st["count"] >= DAILY_QUOTA_PER_KEY:
            logger.warning(f"Key {label} has reached daily quota ({st['count']}/{DAILY_QUOTA_PER_KEY})")
            return False
        
        # Rule 3: Check token bucket
        self._refill_tokens(label)
        if st["tokens"] < 1:
            logger.debug(f"Key {label} has insufficient tokens ({st['tokens']:.2f})")
            return False
        
        return True

    def _get_next_key(self, wait_if_none: bool = False, wait_seconds: int = 3):
        """Thread-safe method to get next API key from rotation cycle with cooldown awareness.

        Returns (api_key, label) or (None, None) when no key is currently usable.
        If wait_if_none=True, it will sleep `wait_seconds` then try once more.
        """
        with self.key_lock:
            for _ in range(len(self.api_keys)):
                api_key, label = next(self.key_cycle)
                if self._can_use_key(label):
                    return api_key, label

        # No immediately usable key
        if wait_if_none:
            time.sleep(wait_seconds)
            with self.key_lock:
                for _ in range(len(self.api_keys)):
                    api_key, label = next(self.key_cycle)
                    if self._can_use_key(label):
                        return api_key, label

        # Explicitly return None to signal caller there is no key available now
        return None, None


    def _should_validate_keys(self) -> bool:
        """Check if keys should be validated using persistent cache"""
        # Check if we have a recent validation cache
        if self._validation_cache_path.exists():
            cache_mtime = self._validation_cache_path.stat().st_mtime
            if time.time() - cache_mtime < self._validation_cache_duration.total_seconds():
                logger.debug("Using recent validation cache, skipping validation")
                return False
        
        logger.debug("Validation cache expired or missing, will validate keys")
        return True

    def _load_validation_cache(self) -> Dict[str, Any]:
        """Load validation cache from disk"""
        if not self._validation_cache_path.exists():
            return {}
        
        try:
            with open(self._validation_cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load validation cache: {e}")
            return {}

    def _save_validation_cache(self, cache_data: Dict[str, Any]):
        """Save validation cache to disk"""
        try:
            cache_data["_metadata"] = {
                "last_validation": datetime.now().isoformat(),
                "cache_version": "1.0",
                "model": self.model
            }
            
            with open(self._validation_cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Validation cache saved to {self._validation_cache_path}")
        except Exception as e:
            logger.error(f"Failed to save validation cache: {e}")

    def _get_key_hash(self, api_key: str) -> str:
        """Create a hash for the API key for caching (for security, don't store full keys)"""
        import hashlib
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]

    def _validate_key(self, api_key: str, label: str) -> bool:
        """Validate API key without consuming quota using metadata check"""
        try:
            genai.configure(api_key=api_key)
            # Use get_model to check key validity without generating content
            model_info = genai.get_model(self.model)
            return model_info is not None and hasattr(model_info, 'name')
        except Exception as e:
            logger.warning(f"Key validation failed for {label}: {str(e)[:100]}")
            return False

    def validate_all_keys(self, force: bool = False) -> List[str]:
        """Validate all API keys and return working ones without consuming quota"""
        working_keys = []
        working_labels = []
        
        # Load existing cache
        cache = self._load_validation_cache()
        key_status_cache = cache.get("key_status", {})
        
        logger.info("Validating API keys (quota-free check)...")
        
        # keep original total for clearer logging later
        orig_total_keys = len(self.api_keys)
        
        for api_key, label in zip(self.api_keys, self.labels):
            key_hash = self._get_key_hash(api_key)
            cache_key = f"{key_hash}_{self.model}"
            
            # Check cache first unless forcing revalidation
            if not force and cache_key in key_status_cache:
                cached_data = key_status_cache[cache_key]
                # safe-guard: cached_data may be malformed; handle gracefully
                try:
                    cached_time = datetime.fromisoformat(cached_data.get("timestamp", "1970-01-01T00:00:00"))
                except Exception:
                    cached_time = datetime.fromtimestamp(0)
                
                if datetime.now() - cached_time < self._validation_cache_duration:
                    is_valid = cached_data["valid"]
                    status = "VALID" if is_valid else "INVALID"
                    logger.info(f"   {'✅' if is_valid else '❌'} {label}: {status} (cached)")
                    
                    if is_valid:
                        working_keys.append(api_key)
                        working_labels.append(label)
                    continue
            
            # Not in cache or cache expired, validate live
            is_valid = self._validate_key(api_key, label)
            
            # Update cache
            key_status_cache[cache_key] = {
                "valid": is_valid,
                "timestamp": datetime.now().isoformat(),
                "label": label,
                "model": self.model
            }
            
            if is_valid:
                working_keys.append(api_key)
                working_labels.append(label)
                logger.info(f"   ✅ {label}: VALID")
            else:
                logger.warning(f"   ❌ {label}: INVALID")
        
        # Save updated cache
        cache["key_status"] = key_status_cache
        self._save_validation_cache(cache)
        
        if not working_keys:
            raise RuntimeError("No valid API keys found!")
        
        # Update with working keys only
        self.api_keys = working_keys
        self.labels = working_labels
        self.key_cycle = cycle(zip(self.api_keys, self.labels))
        
        # FIX: Rebuild key_cooldowns to only include remaining labels
        # keep existing cooldowns where possible, default to 0
        self.key_cooldowns = {label: self.key_cooldowns.get(label, 0) for label in self.labels}
        
        # Update key_usage_state to only include remaining labels
        new_key_usage_state = {}
        today = datetime.now().strftime("%Y-%m-%d")
        for label in self.labels:
            if label in self.key_usage_state:
                new_key_usage_state[label] = self.key_usage_state[label]
                # Reset if it's a new day
                if new_key_usage_state[label]["date"] != today:
                    new_key_usage_state[label] = {
                        "date": today,
                        "count": 0,
                        "tokens": BURST,
                        "last_refill": time.time(),
                        "cooldown_until": 0
                    }
            else:
                new_key_usage_state[label] = {
                    "date": today,
                    "count": 0,
                    "tokens": BURST,
                    "last_refill": time.time(),
                    "cooldown_until": 0
                }
        self.key_usage_state = new_key_usage_state
        self._save_key_usage()
        
        # log using original total so message is not confusing
        logger.info(f"Using {len(working_keys)} valid keys out of {orig_total_keys} total")
        return working_keys

    def _process_single_request(self, prompt: str, timeout_seconds: int = PER_REQUEST_TIMEOUT, **kwargs) -> Optional[Dict[str, Any]]:
        """Central helper for processing single requests with timeout support and robust exception handling"""
        api_key, label = self._get_next_key(wait_if_none=False)

        # If no key is available right now, return a machine-readable error so callers can stop retries
        if api_key is None:
            logger.warning("No API key available: all keys exhausted or cooling down.")
            return {
                "text": "",
                "api_key_label": "none",
                "model_used": self.model,
                "processing_time": 0,
                "error": "no_key_available",
                "exception": None,
                "attempts": 0
            }

        # Proceed as before with a usable api_key,label
        try:
            genai.configure(api_key=api_key)
            model_obj = genai.GenerativeModel(self.model)
            
            # Try to pass timeout via request_options if SDK supports it
            request_kwargs = kwargs.copy()
            # Merge with any existing request_options
            if "request_options" in request_kwargs:
                request_kwargs["request_options"]["timeout"] = timeout_seconds
            else:
                request_kwargs["request_options"] = {"timeout": timeout_seconds}
            
            start_time = time.time()
            response = model_obj.generate_content(prompt, **request_kwargs)
            processing_time = time.time() - start_time
            
            # FREE TIER RATE LIMIT - Enforce ~15 requests per minute per key
            time.sleep(1.2)   # Added to prevent 429 errors
            
            # Thread-safe statistics update
            with self.stats_lock:
                self.stats["total_requests"] += 1
                self.stats["successful_requests"] += 1
                self.stats["key_usage"][label] += 1
                self.stats["total_processing_time"] += processing_time
            
            # Update key usage state on successful request
            with self.key_lock:
                if label in self.key_usage_state:
                    st = self.key_usage_state[label]
                    st["count"] += 1
                    st["tokens"] -= 1
                    self._save_key_usage()
            
            # Safe text extraction with fallback
            response_text = getattr(response, "text", str(response))
            
            return {
                "text": response_text,
                "api_key_label": label,
                "model_used": self.model,
                "processing_time": processing_time,
                "attempts": 1,
                "error": None
            }
            
        except Exception as e:
            # Thread-safe statistics update for failures
            with self.stats_lock:
                self.stats["total_requests"] += 1
                self.stats["failed_requests"] += 1
            
            # Improved 429 detection
            is_429 = False
            if hasattr(e, "status_code") and e.status_code == 429:
                is_429 = True
            elif hasattr(e, "status") and e.status == 429:
                is_429 = True
            elif "429" in str(e):
                is_429 = True
            
            if is_429:
                # Thread-safe cooldown update in key_usage_state
                with self.key_lock:
                    if label in self.key_usage_state:
                        st = self.key_usage_state[label]
                        st["cooldown_until"] = time.time() + COOLDOWN_ON_429
                        self._save_key_usage()
                logger.warning(f"429 hit on {label}, cooling down key for {COOLDOWN_ON_429}s...")
            
            logger.error(f"Error using {label}: {str(e)[:400]}")
            
            return {
                "text": "",
                "api_key_label": label,
                "model_used": self.model,
                "processing_time": 0,
                "error": "exception",
                "exception": str(e),
                "attempts": 0
            }

    def _log_to_sqlite(self, prompt: str, response: Dict[str, Any], label: str):
        """Log request to SQLite database for auditability"""
        try:
            conn = sqlite3.connect(SQLITE_DB)
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS request_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        api_key_label TEXT,
                        model TEXT,
                        prompt TEXT,
                        response_text TEXT,
                        processing_time REAL,
                        attempts INTEGER,
                        error TEXT
                    )
                """)
                # Fixed: Explicit column names to avoid count mismatch
                conn.execute("""
                    INSERT INTO request_logs 
                    (timestamp, api_key_label, model, prompt, response_text, processing_time, attempts, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.utcnow().isoformat() + "Z",
                    label,
                    self.model,
                    prompt[:1000],  # Limit prompt length
                    response["text"][:2000] if response.get("text") else "",  # Limit response length
                    response["processing_time"],
                    response["attempts"],
                    response.get("error")  # Store error if present
                ))
            logger.debug(f"Logged request to SQLite for key {label}")
        except Exception as e:
            logger.error(f"Failed to log to SQLite: {e}")

    def _log_to_json(self, entry: Dict[str, Any]):
        """Log interaction to JSON file for completeness"""
        try:
            with open(JSON_LOG, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.debug("Logged interaction to JSON file")
        except Exception as e:
            logger.error(f"Failed to log to JSON: {e}")

    def generate_single(self, prompt: str, max_retries: int = 3, **kwargs) -> Dict[str, Any]:
        """Generate content for a single prompt with key rotation and retries"""
        if not self.stats["start_time"]:
            self.stats["start_time"] = time.time()
        
        last_error = None
        for attempt in range(max_retries):
            result = self._process_single_request(prompt, PER_REQUEST_TIMEOUT, **kwargs)
            
            if result and not result.get("error"):
                # Log successful request
                self._log_to_sqlite(prompt, result, result["api_key_label"])
                return result
            
            # Immediate abort if no key available (don't waste retries)
            if result and result.get("error") == "no_key_available":
                logger.error("All API keys exhausted / cooling down — aborting further retries.")
                return {
                    "text": "",
                    "api_key_label": result.get("api_key_label", "none"),
                    "model_used": self.model,
                    "processing_time": 0,
                    "error": "no_key_available",
                    "attempts": attempt + 1
                }
            
            # update last_error so the final return message is informative
            last_error = result.get("error") or result.get("exception") or last_error
            # Wait before retry
            backoff = 2 ** attempt
            logger.info(f"Retry {attempt + 1}/{max_retries} after {backoff}s...")
            time.sleep(backoff)
        
        # All retries failed
        return {
            "text": "",
            "api_key_label": "unknown",
            "model_used": self.model,
            "processing_time": 0,
            "error": f"All retries exhausted. Last error: {last_error}",
            "attempts": max_retries
        }

    def run_batch(self, prompts: List[str], max_workers: int = None) -> List[Dict[str, Any]]:
        """Process multiple prompts in parallel with comprehensive timeout controls"""
        if not self.stats["start_time"]:
            self.stats["start_time"] = time.time()
        
        workers = max(1, min(len(prompts), max_workers or self.max_workers))
        results = []
        futures = []
        future_to_prompt = {}
        
        logger.info(f"Starting parallel processing of {len(prompts)} prompts with {workers} workers")
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for prompt in prompts:
                fut = executor.submit(self._process_single_request, prompt, PER_REQUEST_TIMEOUT)
                futures.append(fut)
                future_to_prompt[fut] = prompt

            # Wait for futures with controlled timeouts
            try:
                for future in as_completed(futures, timeout=OVERALL_BATCH_TIMEOUT):
                    try:
                        # per-future timeout ensures a single slow call doesn't block forever
                        res = future.result(timeout=FUTURE_RESULT_TIMEOUT)
                    except TimeoutError:
                        # mark timed-out future as failed and continue
                        logger.error(f"A future timed out after {FUTURE_RESULT_TIMEOUT} seconds")
                        try:
                            # best-effort cancel
                            future.cancel()
                        except Exception:
                            pass
                        results.append({
                            "text": "",
                            "api_key_label": "unknown",
                            "model_used": self.model,
                            "processing_time": 0,
                            "error": "timeout",
                            "attempts": 0
                        })
                        continue
                    except Exception as e:
                        logger.exception(f"Exception while resolving future: {e}")
                        results.append({
                            "text": "",
                            "api_key_label": "unknown",
                            "model_used": self.model,
                            "processing_time": 0,
                            "error": "exception",
                            "exception": str(e)
                        })
                        continue

                    # Normal success path
                    if res and not res.get("error"):
                        # Log successful request
                        prompt_text = future_to_prompt.get(future, "unknown")
                        self._log_to_sqlite(prompt_text, res, res["api_key_label"])
                    results.append(res)
                    
                    # Check for no_key_available error and abort batch if needed
                    if res and res.get("error") == "no_key_available":
                        logger.error("Batch aborted: no API keys available for the remaining prompts.")
                        # mark remaining prompts as failed to avoid waiting on them
                        for fut in futures:
                            if not fut.done():
                                try:
                                    fut.cancel()
                                except Exception:
                                    pass
                                results.append({
                                    "text": "",
                                    "api_key_label": "none",
                                    "model_used": self.model,
                                    "processing_time": 0,
                                    "error": "no_key_available_batch",
                                    "attempts": 0
                                })
                        break
                    
            except TimeoutError:
                # overall as_completed timed out
                logger.error(f"Overall batch wait timed out after {OVERALL_BATCH_TIMEOUT} seconds")
                # For any futures still running, attempt to cancel and mark as timed out
                for fut in futures:
                    if not fut.done():
                        try:
                            fut.cancel()
                        except Exception:
                            pass
                        results.append({
                            "text": "",
                            "api_key_label": "unknown",
                            "model_used": self.model,
                            "processing_time": 0,
                            "error": "overall_batch_timeout"
                        })
        
        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics (thread-safe)"""
        with self.stats_lock:
            total_time = time.time() - self.stats["start_time"] if self.stats["start_time"] else 0
            requests_per_second = self.stats["total_requests"] / total_time if total_time > 0 else 0
            
            stats = {
                "total_requests": self.stats["total_requests"],
                "successful_requests": self.stats["successful_requests"],
                "failed_requests": self.stats["failed_requests"],
                "success_rate": (self.stats["successful_requests"] / self.stats["total_requests"] * 100) if self.stats["total_requests"] > 0 else 0,
                "total_processing_time": self.stats["total_processing_time"],
                "avg_processing_time": (self.stats["total_processing_time"] / self.stats["successful_requests"]) if self.stats["successful_requests"] > 0 else 0,
                "requests_per_second": requests_per_second,
                "key_usage": self.stats["key_usage"].copy(),  # Return a copy for thread safety
                "available_keys": len(self.api_keys),
                "max_workers": self.max_workers,
                "model": self.model
            }
            
            return stats

    def print_real_time_stats(self):
        """Print real-time performance statistics (thread-safe)"""
        stats = self.get_performance_stats()
        
        # Use simple logging without emojis for headless environments
        logger.info("HIGH-SPEED ENGINE STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total Requests: {stats['total_requests']}")
        logger.info(f"Successful: {stats['successful_requests']}")
        logger.info(f"Failed: {stats['failed_requests']}")
        logger.info(f"Success Rate: {stats['success_rate']:.1f}%")
        logger.info(f"Avg Processing Time: {stats['avg_processing_time']:.2f}s")
        logger.info(f"Speed: {stats['requests_per_second']:.2f} requests/second")
        logger.info(f"Available Keys: {stats['available_keys']}")
        logger.info(f"Max Workers: {stats['max_workers']}")
        logger.info(f"Model: {stats['model']}")
        
        logger.info("KEY USAGE:")
        for label, count in stats["key_usage"].items():
            logger.info(f"   {label}: {count} requests")
        
        # Print key usage state
        logger.info("KEY RATE LIMIT STATE:")
        today = datetime.now().strftime("%Y-%m-%d")
        for label in self.labels:
            if label in self.key_usage_state:
                st = self.key_usage_state[label]
                remaining = max(0, DAILY_QUOTA_PER_KEY - st["count"])
                cooldown_msg = ""
                if st["cooldown_until"] > time.time():
                    remaining_cooldown = st["cooldown_until"] - time.time()
                    cooldown_msg = f" [COOLDOWN: {remaining_cooldown:.0f}s]"
                logger.info(f"   {label}: {st['count']}/{DAILY_QUOTA_PER_KEY} daily, {st['tokens']:.1f} tokens{cooldown_msg}")

# ========================
# Flexible Key Manager
# ========================
class FlexibleKeyManager:
    """Manager for flexible key usage patterns"""
    
    @staticmethod
    def get_keys_by_mode(config: Dict[str, Any], keyuse: str = None, speed: str = None) -> tuple:
        """
        Get keys based on usage mode:
        - keyuse: "1" (use only first key), "2" (use first 2 keys), etc.
        - speed: "max" (use all keys in parallel), "3" (use 3 keys in parallel), etc.
        """
        api_keys = config.get("google_api_keys", [])
        labels = config.get("api_key_labels", [f"key{i}" for i in range(len(api_keys))])
        
        if not api_keys:
            return [], []
        
        # Key usage mode
        if keyuse:
            if keyuse == "1":
                # Use only first key
                return [api_keys[0]], [labels[0]]
            else:
                # Use first N keys
                try:
                    n = int(keyuse)
                    return api_keys[:n], labels[:n]
                except ValueError:
                    logger.warning(f"Invalid keyuse value: {keyuse}. Using all keys.")
        
        # Speed mode
        if speed:
            if speed == "max":
                # Use all keys in parallel
                return api_keys, labels
            else:
                # Use first N keys in parallel
                try:
                    n = int(speed)  # FIX: Removed stray colon
                    return api_keys[:n], labels[:n]
                except ValueError:
                    logger.warning(f"Invalid speed value: {speed}. Using all keys.")
        
        # Default: use all keys with rotation
        return api_keys, labels

    @staticmethod
    def calculate_optimal_workers(api_keys: List[str], speed_mode: str = None) -> int:
        """Calculate optimal number of workers based on key count and speed mode"""
        key_count = len(api_keys)
        
        if speed_mode == "max":
            return min(key_count * 3, 16)  # Aggressive parallelization
        elif speed_mode and speed_mode.isdigit():
            workers = int(speed_mode)
            return min(workers * 2, 16)  # Moderate parallelization
        else:
            # FIX: More reasonable worker calculation
            return max(1, min(key_count * 2, 16))  # Conservative parallelization

# ========================
# Enhanced Multi-Key Gemini Client (Backward Compatible)
# ========================
class GeminiMultiKey:
    def __init__(self, api_keys: List[str], labels: Optional[List[str]] = None, 
                 model: str = DEFAULT_MODEL, speed_mode: str = None,
                 validate_keys: bool = False):
        if not HAS_GENAI:
            raise RuntimeError("GenAI SDK not found. Install 'google-generativeai'.")
        
        self.api_keys = api_keys or []
        if not self.api_keys:
            raise RuntimeError("No API keys provided.")
        
        self.labels = labels or [f"key{i}" for i in range(len(self.api_keys))]
        self.model = model
        
        # Initialize high-speed engine
        max_workers = FlexibleKeyManager.calculate_optimal_workers(api_keys, speed_mode)
        self.engine = HighSpeedMultiKeyEngine(
            api_keys, labels, model, max_workers, validate_keys=validate_keys
        )
        
        # Backward compatibility
        self.index = 0

    def _get_next_key(self):
        """Backward compatible key rotation"""
        idx = self.index % len(self.api_keys)
        key = self.api_keys[idx]
        label = self.labels[idx]
        self.index += 1
        return key, label, idx

    def generate(self, prompt: str, files: Optional[List[str]] = None, 
                 images: Optional[List[str]] = None, max_output_tokens: int = 512, 
                 max_retries: int = 3) -> Dict[str, Any]:
        """Enhanced generate method with high-speed capabilities"""
        # Use high-speed engine for single generation
        result = self.engine.generate_single(prompt, max_retries)
        
        # Return complete result with all expected fields
        return {
            "text": result.get("text", ""),
            "raw": {"text": result.get("text", "")},
            "api_key_label": result.get("api_key_label", "unknown"),
            "model_used": result.get("model_used", self.model),
            "processing_time": result.get("processing_time", 0),
            "attempts": result.get("attempts", 1),
            "error": result.get("error")
        }

    def generate_batch(self, prompts: List[str], max_workers: int = None) -> List[Dict[str, Any]]:
        """Generate content for multiple prompts in parallel"""
        return self.engine.run_batch(prompts, max_workers)

    def validate_keys(self, force: bool = False) -> List[str]:
        """Validate all API keys"""
        return self.engine.validate_all_keys(force=force)

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.engine.get_performance_stats()

    def print_stats(self):
        """Print performance statistics"""
        self.engine.print_real_time_stats()

# ========================
# File Extraction (Unchanged)
# ========================
def extract_text_from_file(path: str) -> str:
    p = pathlib.Path(path)
    if not p.exists():
        return ""
    if p.suffix.lower() in ['.txt', '.md', '.csv', '.log']:
        return p.read_text(encoding='utf-8', errors='ignore')
    if p.suffix.lower() == '.pdf':
        text = []
        with open(p, 'rb') as fh:
            reader = PdfReader(fh)
            for page in reader.pages:
                try:
                    text.append(page.extract_text() or "")
                except Exception:
                    continue
        return "\n".join(text)
    return f"[Non-text file: {p.name} — not extracted]"

# ========================
# Enhanced High-Level Request Flow
# ========================
def process_request(
    gemini_client: GeminiMultiKey,
    prompt: str,
    file_paths: Optional[List[str]] = None,
    image_paths: Optional[List[str]] = None,
    do_web_search: bool = False,
    search_config: Optional[dict] = None,
    db_conn=None
) -> Dict[str, Any]:
    logger.info("Processing new request...")
    file_paths = file_paths or []
    image_paths = image_paths or []

    # Extract file texts
    file_texts = []
    for p in file_paths:
        txt = extract_text_from_file(p)
        file_texts.append({"path": p, "text_excerpt": txt[:2000]})
    logger.debug(f"Attached files: {[p['path'] for p in file_texts]}")

    # Web search removed as requested
    search_snippet = ""
    if do_web_search:
        logger.warning("Web search functionality has been removed from this version")

    composed = prompt
    if file_texts:
        composed += "\n\n--- Attached files (first 2k chars each) ---\n"
        for f in file_texts:
            composed += f"\nFile: {f['path']}\n{f['text_excerpt']}\n"
    if search_snippet:
        composed += "\n\n--- Web Search Results ---\n" + search_snippet

    # Call Gemini with high-speed engine
    resp = gemini_client.generate(composed, files=[p for p in file_paths], images=image_paths)

    # If helper reports there are no usable keys, abort cleanly and return an informative entry
    if resp.get("error") == "no_key_available":
        logger.error("Aborting request: no API keys available (exhausted or cooling).")
        timestamp = datetime.utcnow().isoformat() + "Z"
        entry = {
            "timestamp": timestamp,
            "api_key_label": resp.get("api_key_label", "none"),
            "model": gemini_client.model,
            "prompt": prompt,
            "web_search": "",
            "files": [str(p) for p in file_paths],
            "response_text": None,
            "raw_response": resp
        }
        # Log to JSON for audit (optional)
        gemini_client.engine._log_to_json(entry)
        # Return an explicit error entry so callers can handle it
        return entry
    
    timestamp = datetime.utcnow().isoformat() + "Z"
    entry = {
        "timestamp": timestamp,
        "api_key_label": resp.get("api_key_label"),
        "model": gemini_client.model,
        "prompt": prompt,
        "web_search": search_snippet,
        "files": [str(p) for p in file_paths],
        "response_text": resp.get("text"),
        "raw_response": resp.get("raw")
    }

    # Save to JSON log for completeness
    gemini_client.engine._log_to_json(entry)

    # Save to database if connection provided
    if db_conn:
        try:
            cur = db_conn.cursor()
            cur.execute("""
            INSERT INTO interactions (timestamp, api_key_label, model, prompt, web_search, files, response_text, raw_response_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry["timestamp"],
                entry["api_key_label"],
                entry["model"],
                entry["prompt"],
                entry["web_search"][:4000] if entry["web_search"] else None,
                json.dumps(entry["files"], ensure_ascii=False),
                entry["response_text"][:4000] if entry["response_text"] else None,
                json.dumps(entry["raw_response"], default=str, ensure_ascii=False)
            ))
            db_conn.commit()
            logger.debug("Saved interaction to SQLite DB.")
        except Exception as e:
            logger.error(f"DB insert failed: {e}")

    logger.info("Request processed successfully.")
    return entry

# ========================
# Enhanced CLI Entry
# ========================
def main():
    parser = argparse.ArgumentParser(description="Gemini multi-key requester (HIGH-SPEED ENHANCED)")
    parser.add_argument("--config", default=CONFIG_FILE, help="Path to config file")
    parser.add_argument("--prompt", help="Prompt text (if omitted, read from stdin)")
    parser.add_argument("--files", nargs="*", help="Paths to files to include")
    parser.add_argument("--images", nargs="*", help="Paths to images (optional)")
    parser.add_argument("--web", action="store_true", help="Web search (disabled in this version)")
    parser.add_argument("--model", default=DEFAULT_MODEL, 
                       choices=list(AVAILABLE_MODELS.values()),
                       help="Model to use for generation")
    
    # New high-speed options
    parser.add_argument("--keyuse", help="Key usage: '1' (use only first key), '2' (use first 2 keys), etc.")
    parser.add_argument("--speed", help="Speed mode: 'max' (use all keys), '2' (use 2 keys in parallel), etc.")
    parser.add_argument("--validate-keys", action="store_true", help="Force validate all API keys before use")
    parser.add_argument("--no-validate-keys", action="store_true", help="Disable automatic key validation")
    parser.add_argument("--force-validate", action="store_true", help="Force revalidation ignoring cache")
    parser.add_argument("--batch", nargs="*", help="Process multiple prompts in batch")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()

    # Reconfigure logging based on verbosity
    global logger
    logger = setup_logging(args.verbose)
    
    logger.info(f"Starting HIGH-SPEED Gemini requester — model={args.model}")
    logger.info(f"Log directory: {LOG_DIR}")
    logger.info(f"Data directory: {DATA_DIR}")
    
    cfg = load_config(args.config)
    
    # Use flexible key manager
    api_keys, labels = FlexibleKeyManager.get_keys_by_mode(cfg, args.keyuse, args.speed)
    
    if not api_keys:
        env_keys = os.getenv("GOOGLE_API_KEYS")
        if env_keys:
            api_keys = [k.strip() for k in env_keys.split(",") if k.strip()]
            labels = [f"env_key_{i}" for i in range(len(api_keys))]
    
    if not api_keys:
        logger.critical("No API keys provided; exiting.")
        raise SystemExit("Provide keys in config.json or via GOOGLE_API_KEYS env var.")

    logger.info(f"Loaded {len(api_keys)} API keys: {labels}")
    
    # Determine validation mode
    validate_keys = args.validate_keys or args.force_validate
    if not args.no_validate_keys and not validate_keys:
        # Auto-validate once per day by default (using cache)
        validate_keys = True
    
    # Initialize high-speed client
    gm = GeminiMultiKey(
        api_keys=api_keys, 
        labels=labels, 
        model=args.model, 
        speed_mode=args.speed,
        validate_keys=validate_keys
    )
    
    # Force validate keys if explicitly requested (with optional cache bypass)
    if args.validate_keys or args.force_validate:
        working_keys = gm.validate_keys(force=args.force_validate)
        logger.info(f"Using {len(working_keys)} validated keys")

    # Batch processing
    if args.batch:
        logger.info(f"Processing {len(args.batch)} prompts in batch mode...")
        results = gm.generate_batch(args.batch)
        
        # If any result returned 'no_key_available', then abort remaining processing and mark them
        if any(r.get("error") == "no_key_available" for r in results):
            logger.error("Batch aborted early: one or more requests hit 'no_key_available'. Marking remaining prompts as failed.")
            # find index of first occurrence (so we can align failure markers with prompts)
            first_idx = next((i for i, r in enumerate(results) if r.get("error") == "no_key_available"), -1)
            # Expand results to equal length of prompts (if futures were cancelled there may be fewer)
            while len(results) < len(args.batch):
                results.append({
                    "text": "",
                    "api_key_label": "none",
                    "model_used": gm.model,
                    "processing_time": 0,
                    "error": "no_key_available_batch",
                    "attempts": 0
                })
        
        # summarize batch failures/successes
        successes = sum(1 for r in results if r.get("error") is None)
        failures = len(results) - successes
        print(f"\nBatch processed: successes={successes}, failures={failures}")
        if failures > 0:
            print("⚠️ Some requests failed in this batch — check logs/gemini_multikey.log for details.")
        
        for i, (prompt, result) in enumerate(zip(args.batch, results)):
            print(f"\n---- RESPONSE {i+1} ----")
            print(f"Prompt: {prompt[:100]}...")
            print(f"Response: {result['text']}")
            if result.get('error'):
                print(f"Error: {result['error']}")
        
        gm.print_stats()
        
    else:
        # Single prompt processing
        prompt = args.prompt or input("Enter prompt: ").strip()
        
        # Web search is disabled but kept for backward compatibility
        search_cfg = {}
        if args.web:
            logger.warning("Web search is disabled in this version")

        try:
            entry = process_request(
                gemini_client=gm,
                prompt=prompt,
                file_paths=args.files,
                image_paths=args.images,
                do_web_search=args.web,
                search_config=search_cfg,
                db_conn=None  # Disable database logging for CLI
            )
            print("\n---- RESPONSE ----\n")
            print(entry["response_text"] or json.dumps(entry["raw_response"], indent=2))
            print(f"\nProcessed using {entry['api_key_label']}")
            
            gm.print_stats()
            logger.info("CLI execution completed.")
            
        except Exception as e:
            logger.exception(f"Fatal error during request: {e}")
            raise SystemExit(1)

if __name__ == "__main__":
    main()