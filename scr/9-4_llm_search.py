#!/usr/bin/env python3
"""
9-4_llm_search.py - Enhanced LLM Job Analysis with Database
Purpose: Analyze shortlisted jobs using Gemini AI with SQLite storage
Integrated with gemini_multikey_9_3_helper_script.py for multi-key support
"""

import json
import time
import sqlite3
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import math
import sys
import os
import logging
import inspect
from logging.handlers import RotatingFileHandler
import concurrent.futures

# Add the parent directory to Python path to import the helper script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import the multi-key Gemini client from your helper script
try:
    from gemini_multikey_9_3_helper_script import GeminiMultiKey, load_config, FlexibleKeyManager
    HAS_MULTIKEY = True
except ImportError as e:
    HAS_MULTIKEY = False
    # We'll define a basic fallback implementation

# ───── Helper Function for Key Exhaustion Detection ─────
def _is_no_key_available(result):
    """Return True if the result signals that all API keys are exhausted / cooling."""
    if not isinstance(result, dict):
        return False
    err = result.get("error")
    if err == "no_key_available":
        return True
    # some clients may put a message string in gemini_response — keep a defensive check
    gem_resp = result.get("gemini_response", "") or ""
    if isinstance(gem_resp, str) and "All API keys exhausted" in gem_resp:
        return True
    return False

# ───── Configuration ─────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
RESULT_DIR = DATA_DIR / "llm_results"
SHORTLIST_PATH = DATA_DIR / "shortlisted_jobs_json" / "shortlisted_jobs.json"
RESUME_DIR = DATA_DIR / "dynamic_resumes"
RESULT_JSON = RESULT_DIR / f"llm_job_analysis_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.json"
RESULT_DB = RESULT_DIR / "llm_job_analysis.db"
CONFIG_FILE = BASE_DIR / "gemini_config.json"
PROGRESS_FILE = RESULT_DIR / "progress.json"

# Batch processing configuration - LOWERED FOR DEBUGGING
BATCH_SIZE = 2  # Reduced from 8 to 2 for debugging
MIN_BATCH_SIZE = 2  # Minimum batch size for parallel processing
PER_REQUEST_TIMEOUT = 60  # per-request timeout (seconds)

# Create directories
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ───── Enhanced Logging Setup ─────
def setup_main_logging():
    """Setup professional logging for the main script - minimal console, detailed file"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"llm_job_analysis_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler - DETAILED (everything)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Setup console handler - MINIMAL (only warnings and errors)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)  # Only show warnings and errors in console
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Also configure the gemini_multikey logger if available - WITH CONSOLE FOR WARNINGS/ERRORS
    try:
        gemini_logger = logging.getLogger("gemini_multikey")
        gemini_logger.setLevel(logging.DEBUG)
        for handler in gemini_logger.handlers[:]:
            gemini_logger.removeHandler(handler)
        gemini_logger.addHandler(file_handler)  # Detailed in file
        
        # ADD CONSOLE HANDLER FOR WARNINGS AND ERRORS
        console_handler_gemini = logging.StreamHandler()
        console_handler_gemini.setLevel(logging.WARNING)
        console_handler_gemini.setFormatter(formatter)
        gemini_logger.addHandler(console_handler_gemini)
        
        gemini_logger.propagate = False  # Prevent propagation to root logger
    except Exception as e:
        logging.error(f"Failed to set up gemini_multikey logger: {e}")
    
    logging.info(f"📝 Logging initialized: {log_file}")
    return log_file

# ───── Basic Gemini Fallback Implementation ─────
if not HAS_MULTIKEY:
    import google.generativeai as genai
    from tenacity import retry, wait_exponential, stop_after_attempt
    
    class BasicGeminiClient:
        def __init__(self, api_key):
            self.api_key = api_key
            genai.configure(api_key=api_key)
            self.model_name = "gemini-2.5-flash"
            self.model = genai.GenerativeModel(self.model_name)
            self.stats = {"requests": 0, "failures": 0}
            # Key-level cooldowns for rate limiting
            self.key_cooldowns = {"basic_key": 0}
        
        @retry(
            wait=wait_exponential(multiplier=1, min=2, max=30),
            stop=stop_after_attempt(3),
            reraise=True
        )
        def generate(self, prompt, max_retries=3):
            """Basic generate method for fallback"""
            # Check if key is in cooldown
            cooldown_until = self.key_cooldowns.get("basic_key", 0)
            current_time = time.time()
            if current_time < cooldown_until:
                wait_time = cooldown_until - current_time
                logging.warning(f"Key in cooldown, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            
            self.stats["requests"] += 1
            try:
                start_time = time.time()
                response = self.model.generate_content(prompt)
                processing_time = time.time() - start_time
                
                return {
                    "text": response.text,
                    "api_key_label": "basic_key",
                    "model_used": self.model_name,
                    "processing_time": processing_time,
                    "attempts": 1,
                    "error": None  # Add error field for consistency
                }
            except Exception as e:
                self.stats["failures"] += 1
                # Check if it's a rate limit error (429)
                if "429" in str(e):
                    # Mark this key as in cooldown for 60 seconds
                    self.key_cooldowns["basic_key"] = time.time() + 60
                    logging.warning("Rate limit hit (429), key cooling down for 60s")
                    # Return explicit no_key_available for single-key mode too
                    return {
                        "text": "ERROR: All API keys exhausted or cooling down",
                        "api_key_label": "basic_key",
                        "model_used": self.model_name,
                        "processing_time": 0,
                        "attempts": 3,
                        "error": "no_key_available"
                    }
                return {
                    "text": f"ERROR: {str(e)}",
                    "api_key_label": "basic_key",
                    "model_used": self.model_name,
                    "processing_time": 0,
                    "attempts": 3,
                    "error": str(e)
                }
        
        def _get_next_key(self):
            """Simple key selection for single-key client"""
            cooldown_until = self.key_cooldowns.get("basic_key", 0)
            if time.time() < cooldown_until:
                wait_time = cooldown_until - time.time()
                if wait_time > 0:
                    logging.info(f"Waiting {wait_time:.1f}s for key cooldown")
                    time.sleep(wait_time)
            return self.api_key, "basic_key"
        
        def validate_keys(self):
            """Basic validation"""
            return [self.api_key]
        
        def print_stats(self):
            """Print basic stats - to console only"""
            print(f"📊 Basic Gemini Client Stats:")
            print(f"   Total Requests: {self.stats['requests']}")
            print(f"   Failures: {self.stats['failures']}")
            if self.stats['requests'] > 0:
                success_rate = (self.stats['requests'] - self.stats['failures']) / self.stats['requests'] * 100
                print(f"   Success Rate: {success_rate:.1f}%")

# ───── Database Functions ─────
def init_db(db_path=RESULT_DB):
    """Initialize SQLite database for storing analysis results"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS job_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL,
        job_text TEXT,
        resume_used TEXT,
        similarity_score REAL,
        gemini_response TEXT,
        analysis_timestamp TEXT,
        status TEXT,
        model_used TEXT,
        application_status TEXT DEFAULT 'pending',
        notes TEXT,
        api_key_label TEXT,
        processing_time REAL,
        attempts INTEGER,
        prompt_length INTEGER,
        response_length INTEGER,
        estimated_prompt_tokens INTEGER,
        estimated_response_tokens INTEGER,
        total_estimated_tokens INTEGER,
        tokens_per_second REAL,
        UNIQUE(job_id)
    )
    """)
    
    # Create indexes for better query performance
    cur.execute("CREATE INDEX IF NOT EXISTS idx_status ON job_analysis(status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_similarity ON job_analysis(similarity_score)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_resume ON job_analysis(resume_used)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_api_key ON job_analysis(api_key_label)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens ON job_analysis(total_estimated_tokens)")
    
    conn.commit()
    logging.info(f"🗄️ Database initialized: {db_path}")
    return conn

def save_to_db(conn, analysis_result):
    """Save analysis result to database with upsert (update if exists)"""
    cur = conn.cursor()
    try:
        cur.execute("""
        INSERT OR REPLACE INTO job_analysis 
        (job_id, job_text, resume_used, similarity_score, gemini_response, 
         analysis_timestamp, status, model_used, application_status, api_key_label,
         processing_time, attempts, prompt_length, response_length, 
         estimated_prompt_tokens, estimated_response_tokens, total_estimated_tokens,
         tokens_per_second)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            analysis_result["job_id"],
            analysis_result["job_text"][:2000],  # Truncate if too long
            analysis_result["resume_used"],
            analysis_result["similarity"],
            analysis_result["gemini_response"],
            analysis_result.get("processed_at", datetime.now().isoformat()),
            analysis_result["status"],
            analysis_result.get("model_used", "unknown"),
            analysis_result.get("application_status", "pending"),
            analysis_result.get("api_key_label", "unknown"),
            analysis_result.get("processing_time", 0),
            analysis_result.get("attempts", 1),
            analysis_result.get("prompt_length", 0),
            analysis_result.get("response_length", 0),
            analysis_result.get("estimated_prompt_tokens", 0),
            analysis_result.get("estimated_response_tokens", 0),
            analysis_result.get("total_estimated_tokens", 0),
            analysis_result.get("tokens_per_second", 0)
        ))
        conn.commit()
        logging.debug(f"💾 Saved to DB: {analysis_result['job_id']}")
        return True
    except Exception as e:
        logging.exception(f"Database save failed for job {analysis_result.get('job_id', 'unknown')}")
        conn.rollback()
        return False

# ───── Token Estimation Functions ─────
def calculate_token_efficiency(prompt_length, response_length, processing_time):
    """Calculate token efficiency metrics."""
    prompt_tokens = prompt_length // 4
    response_tokens = response_length // 4
    total_tokens = prompt_tokens + response_tokens
    
    tokens_per_second = total_tokens / processing_time if processing_time > 0 else 0
    
    return {
        "prompt_length": prompt_length,
        "response_length": response_length,
        "estimated_prompt_tokens": prompt_tokens,
        "estimated_response_tokens": response_tokens,
        "total_estimated_tokens": total_tokens,
        "tokens_per_second": tokens_per_second
    }

# ───── Progress Tracking Functions ─────
def save_progress(current_index, total_jobs=None, results=None):
    """Save progress for resumability"""
    results = results or []
    progress_data = {
        "last_processed": current_index,
        "total_jobs": total_jobs if total_jobs is not None else -1,
        "processed_count": len(results),
        "successful_count": sum(1 for r in results if r["status"] == "success"),
        "timestamp": datetime.now().isoformat(),
        # store only IDs to keep file small
        "processed_jobs": [r["job_id"] for r in results]
    }
    
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)
        logging.debug(f"💾 Progress saved: {current_index}/{total_jobs if total_jobs is not None else '?'} jobs")
    except Exception as e:
        logging.error(f"Failed to save progress: {e}")

def load_progress():
    """Load progress if exists"""
    if not PROGRESS_FILE.exists():
        return None
    
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load progress: {e}")
        return None

def cleanup_progress():
    """Clean up progress file after successful completion"""
    if PROGRESS_FILE.exists():
        try:
            PROGRESS_FILE.unlink()
            logging.info("✅ Progress file cleaned up")
        except Exception as e:
            logging.warning(f"Failed to clean up progress file: {e}")

# ───── Enhanced Prompt Builder ─────
def build_enhanced_prompt(job_entry, resume_text):
    """Build comprehensive analysis prompt"""
    job_text = job_entry["job_text"]
    job_id = job_entry["job_id"]
    newspaper = job_id.split("/")[0] if "/" in job_id else "Unknown"
    similarity = job_entry.get("similarity", 0.0)

    prompt = f"""
JOB ANALYSIS REQUEST

JOB POSTING:
- Source: {newspaper}
- Similarity Score: {similarity:.3f}
- Content: "{job_text}"

CANDIDATE BACKGROUND:
{resume_text[:3500]}

REQUIRED ANALYSIS:

1. JOB IDENTIFICATION
   - Probable job title and employer
   - Industry/sector classification
   - Key responsibilities inferred

2. SKILLS & QUALIFICATIONS MATCH
   - Required skills from job posting
   - Candidate's matching competencies
   - Notable gaps or mismatches
   - Experience alignment

3. RELEVANCE ASSESSMENT
   - Overall match score (1-5, where 5 is perfect)
   - Detailed justification for scoring
   - Key strengths in this match
   - Potential concerns

4. RECOMMENDATION & STRATEGY
   - Clear verdict: "RECOMMENDED" or "NOT RECOMMENDED"
   - Primary reasons (3 bullet points maximum)
   - If recommended: specific application strategy
   - If not recommended: alternative suggestions

5. ADDITIONAL INSIGHTS
   - Estimated competitiveness level
   - Suggested resume tweaks for this role
   - Potential interview focus areas

RESPONSE FORMAT:
- Use clear section headings
- Be specific and evidence-based
- Focus on actionable insights
- Avoid generic statements
- Keep under 400 words
- Use bullet points for readability
"""
    return prompt.strip()

# ───── File Extraction Helper ─────
def extract_text_from_file(path: str) -> str:
    """Extract text from various file types"""
    p = Path(path)
    if not p.exists():
        return ""
    if p.suffix.lower() in ['.txt', '.md', '.csv', '.log']:
        return p.read_text(encoding='utf-8', errors='ignore')
    return f"[File: {p.name} — content loaded]"

# ───── Batch Processing Functions ─────
def process_job_batch(gemini_client, job_batch, resume_dir, batch_idx=None, total_batches=None):
    """Process a batch of jobs sequentially (no generate_batch)"""
    results = []
    job_metadata = []
    
    # Prepare prompts and metadata
    for job_entry in job_batch:
        resume_path = resume_dir / job_entry["best_resume"]
        if not resume_path.exists():
            logging.warning(f"Resume not found: {resume_path}")
            continue
            
        resume_text = extract_text_from_file(str(resume_path))
        prompt = build_enhanced_prompt(job_entry, resume_text)
        
        job_metadata.append({
            "job_entry": job_entry,
            "prompt": prompt,
            "prompt_length": len(prompt),
            "resume_path": resume_path
        })
    
    if not job_metadata:
        return []
    
    # Process each job sequentially with delay
    success_count = 0
    fail_count = 0
    
    # Print batch info to console
    if batch_idx is not None and total_batches is not None:
        print(f"\n📦 Batch {batch_idx+1}/{total_batches}: Processing {len(job_metadata)} jobs sequentially")
    else:
        print(f"\n📦 Processing {len(job_metadata)} jobs sequentially")
    
    for i, metadata in enumerate(job_metadata, 1):
        job_entry = metadata["job_entry"]
        prompt = metadata["prompt"]
        
        try:
            # Process with Gemini - sequential calls only
            result = gemini_client.generate(prompt)
            
            # Check if all API keys are exhausted
            if _is_no_key_available(result):
                logging.warning("All API keys exhausted or cooling down — stopping batch")
                # Create error result for the current job
                analysis_result = {
                    "job_id": job_entry["job_id"],
                    "job_text": job_entry["job_text"],
                    "resume_used": job_entry["best_resume"],
                    "similarity": job_entry["similarity"],
                    "gemini_response": "ERROR: All API keys exhausted or cooling down",
                    "processed_at": datetime.now().isoformat(),
                    "model_used": gemini_client.model if hasattr(gemini_client, 'model') else "unknown",
                    "api_key_label": "exhausted",
                    "processing_time": 0,
                    "attempts": 1,
                    "status": "error",
                    "application_status": "pending",
                    **calculate_token_efficiency(metadata["prompt_length"], 0, 0)
                }
                results.append(analysis_result)
                # Persist progress immediately for safety
                try:
                    # We don't have total_jobs here, so pass None (will be stored as -1)
                    save_progress(batch_idx * BATCH_SIZE + i - 1, None, results)
                except Exception as e:
                    logging.debug(f"Failed to save progress at exhaustion point: {e}")
                break  # Stop processing further jobs in this batch
            
            # EXTRA safety at caller level - 1.2 second delay between requests
            if i < len(job_metadata):
                time.sleep(1.2)
            
            # Validate result structure
            if not isinstance(result, dict):
                logging.error(f"Result for job {i} is not a dictionary")
                result = {
                    "text": "",
                    "api_key_label": "unknown",
                    "model_used": gemini_client.model if hasattr(gemini_client, 'model') else "unknown",
                    "processing_time": 0,
                    "attempts": 1,
                    "error": "Invalid response format"
                }
            
            response_text = result.get("text", "")
            processing_time = result.get("processing_time", 0)
            api_key_label = result.get("api_key_label", "unknown")
            attempts = result.get("attempts", 1)
            
            # Calculate token efficiency metrics
            token_metrics = calculate_token_efficiency(
                prompt_length=metadata["prompt_length"],
                response_length=len(response_text),
                processing_time=processing_time
            )
            
            # Determine status based on error field or empty response
            if result.get("error") is not None or not response_text:
                status = "error"
                fail_count += 1
            else:
                status = "success"
                success_count += 1
            
            analysis_result = {
                "job_id": job_entry["job_id"],
                "job_text": job_entry["job_text"],
                "resume_used": job_entry["best_resume"],
                "similarity": job_entry["similarity"],
                "gemini_response": response_text,
                "processed_at": datetime.now().isoformat(),
                "model_used": gemini_client.model if hasattr(gemini_client, 'model') else "unknown",
                "api_key_label": api_key_label,
                "processing_time": processing_time,
                "attempts": attempts,
                "status": status,
                "application_status": "pending",
                **token_metrics
            }
            
            results.append(analysis_result)
            
            # Log progress
            if i % 2 == 0 or i == len(job_metadata):
                print(f"   Processed {i}/{len(job_metadata)}: {success_count}✓ {fail_count}✗")
                
        except Exception as e:
            logging.exception(f"Failed to process job {i}: {e}")
            fail_count += 1
            
            # Create error result
            analysis_result = {
                "job_id": job_entry["job_id"],
                "job_text": job_entry["job_text"],
                "resume_used": job_entry["best_resume"],
                "similarity": job_entry["similarity"],
                "gemini_response": f"ERROR: {str(e)}",
                "processed_at": datetime.now().isoformat(),
                "model_used": gemini_client.model if hasattr(gemini_client, 'model') else "unknown",
                "api_key_label": "error",
                "processing_time": 0,
                "attempts": 1,
                "status": "error",
                "application_status": "pending",
                **calculate_token_efficiency(metadata["prompt_length"], 0, 0)
            }
            results.append(analysis_result)
            
            # Continue with next job despite error
            if i < len(job_metadata):
                time.sleep(1.2)
    
    # Print batch summary
    print(f"📊 Batch complete: {success_count}✓ {fail_count}✗")
    return results

def process_single_job(gemini_client, job_entry, resume_dir):
    """Process a single job (fallback for small batches)"""
    resume_path = resume_dir / job_entry["best_resume"]
    if not resume_path.exists():
        logging.warning(f"Resume not found: {resume_path}")
        return None
        
    resume_text = extract_text_from_file(str(resume_path))
    prompt = build_enhanced_prompt(job_entry, resume_text)
    
    # Process with Gemini - sequential call
    try:
        result = gemini_client.generate(prompt)
        
        # Check if all API keys are exhausted
        if _is_no_key_available(result):
            logging.warning("All API keys exhausted or cooling down — stopping")
            return {
                "job_id": job_entry["job_id"],
                "job_text": job_entry["job_text"],
                "resume_used": job_entry["best_resume"],
                "similarity": job_entry["similarity"],
                "gemini_response": "ERROR: All API keys exhausted or cooling down",
                "processed_at": datetime.now().isoformat(),
                "model_used": gemini_client.model if hasattr(gemini_client, 'model') else "unknown",
                "api_key_label": "exhausted",
                "processing_time": 0,
                "attempts": 1,
                "status": "error",
                "application_status": "pending",
                **calculate_token_efficiency(len(prompt), 0, 0)
            }
            
    except Exception as e:
        logging.exception(f"generate() failed for job {job_entry.get('job_id')}: {e}")
        result = {
            "text": "",
            "api_key_label": "unknown",
            "processing_time": 0,
            "attempts": 0,
            "error": str(e)
        }
    
    # Calculate token efficiency metrics
    token_metrics = calculate_token_efficiency(
        prompt_length=len(prompt),
        response_length=len(result.get("text", "")),
        processing_time=result.get("processing_time", 0)
    )
    
    # Determine status
    if result.get("error") is not None or not result.get("text"):
        status = "error"
    else:
        status = "success"
    
    analysis_result = {
        "job_id": job_entry["job_id"],
        "job_text": job_entry["job_text"],
        "resume_used": job_entry["best_resume"],
        "similarity": job_entry["similarity"],
        "gemini_response": result.get("text", ""),
        "processed_at": datetime.now().isoformat(),
        "model_used": gemini_client.model if hasattr(gemini_client, 'model') else "unknown",
        "api_key_label": result.get("api_key_label", "unknown"),
        "processing_time": result.get("processing_time", 0),
        "attempts": result.get("attempts", 1),
        "status": status,
        "application_status": "pending",
        **token_metrics
    }
    
    return analysis_result

# ───── Gemini Client Validation ─────
def validate_gemini_client(gemini_client):
    """Validate that the Gemini client has required methods"""
    required_methods = ['generate']
    
    for method in required_methods:
        if not hasattr(gemini_client, method):
            logging.warning(f"Gemini client missing method '{method}'")
            return False
    
    # generate_batch is no longer required
    logging.info("Using sequential processing (no generate_batch)")
    
    return True

# ───── Enhanced Config Loader ─────
def safe_load_config(config_path):
    """Safely load config with proper error handling"""
    try:
        return load_config(config_path)
    except Exception as e:
        logging.error(f"Error loading config from {config_path}: {e}")
        api_keys = os.getenv("GOOGLE_API_KEYS")
        if api_keys:
            return {
                "google_api_keys": [k.strip() for k in api_keys.split(",") if k.strip()],
                "api_key_labels": [f"env_key_{i}" for i in range(len(api_keys.split(",")))]
            }
        raise

# ───── Cache Cleanup Function ─────
def clear_validation_cache():
    """Clear the validation cache to force fresh validation"""
    cache_file = DATA_DIR / ".key_validation_cache.json"
    if cache_file.exists():
        try:
            cache_file.unlink()
            logging.info("🧹 Cleared validation cache")
        except Exception as e:
            logging.warning(f"Failed to clear cache: {e}")

# ───── Console Print Functions (for clean terminal output) ─────
def print_console(message):
    """Print to console only (not to log file)"""
    print(message)

def print_summary(title, items):
    """Print a formatted summary to console"""
    print_console(f"\n{title}")
    for item in items:
        print_console(f"   {item}")

# ───── Main Execution ─────
def main():
    # Start runtime timer
    start_time = time.time()
    
    # Setup logging FIRST - minimal console, detailed file
    log_file = setup_main_logging()
    
    # CONSOLE: Simple startup message
    print_console("🚀 Starting LLM Job Analysis")
    print_console("=" * 40)
    
    # Load shortlisted jobs
    if not SHORTLIST_PATH.exists():
        print_console(f"❌ Shortlisted jobs file not found: {SHORTLIST_PATH}")
        return
    
    try:
        with open(SHORTLIST_PATH, "r", encoding="utf-8") as f:
            shortlisted = json.load(f)
        
        # Sanity-check the JSON schema
        if not isinstance(shortlisted, list):
            print_console("❌ Shortlisted jobs JSON malformed: expected list of jobs.")
            return
    except json.JSONDecodeError as e:
        print_console(f"❌ Error parsing shortlisted jobs JSON: {e}")
        return
    except Exception as e:
        print_console(f"❌ Error loading shortlisted jobs: {e}")
        return
    
    total_jobs = len(shortlisted)
    print_console(f"📄 Loading {total_jobs} jobs...")
    
    # Clear validation cache
    clear_validation_cache()
    
    # Ensure key usage file exists and is writable so helper can persist counts
    key_usage_file = DATA_DIR / ".key_usage.json"
    try:
        if not key_usage_file.exists():
            key_usage_file.write_text("{}")
        # quick write permission test
        key_usage_file.write_text(key_usage_file.read_text())
    except Exception as e:
        print_console(f"⚠️ Warning: cannot write key usage file {key_usage_file}: {e}. "
                      "This may cause counters to reset between runs.")

    # Load config and initialize Gemini client
    try:
        config = safe_load_config(CONFIG_FILE)
        api_keys = config.get("google_api_keys", [])
        
        if not api_keys:
            raise Exception("No API keys found in config")
        
        # Initialize appropriate Gemini client
        if HAS_MULTIKEY:
            labels = config.get("api_key_labels", [])
            gemini_client = GeminiMultiKey(
                api_keys=api_keys,
                labels=labels,
                model="gemini-2.5-flash",
                speed_mode=None  # conservative; we call generate() sequentially
            )
        else:
            gemini_client = BasicGeminiClient(api_keys[0])
        
        # Validate keys (support clients that may not accept a 'force' arg)
        try:
            working_keys = gemini_client.validate_keys(force=True)
        except TypeError:
            working_keys = gemini_client.validate_keys()
        
        # Normalize the returned value to a list and sanity-check
        if isinstance(working_keys, (str,)):
            working_keys = [working_keys]
        if not working_keys:
            logging.critical("No working API keys found after validation — aborting run.")
            print_console("❌ No working API keys found after validation. Aborting.")
            return

        # Warn if fewer working keys than configured (helps explain unexpected failures)
        if len(working_keys) < len(api_keys):
            print_console(f"⚠️ Fewer validated keys ({len(working_keys)}) than configured ({len(api_keys)}). "
                          "This can increase failure rate if you expected more parallel calls.")
        print_console(f"🔑 Using {len(working_keys)} API key(s)")
        
    except Exception as e:
        print_console(f"❌ Gemini client initialization failed: {e}")
        return

    # Initialize database
    try:
        db_conn = init_db()
    except Exception as e:
        print_console(f"❌ Database initialization failed: {e}")
        return

    # Resume functionality
    progress = load_progress()
    start_index = 0
    results = []
    
    if progress:
        start_index = progress["last_processed"] + 1
        # We don't restore full results from progress, just the index
        previously_processed = progress.get("successful_count", 0)
        processed_jobs = progress.get("processed_jobs", [])
        print_console(f"🔄 Resuming from job {start_index + 1}/{total_jobs}")
        print_console(f"📊 Previously processed: {previously_processed} successful jobs")
    else:
        print_console("🆕 Starting new analysis")

    # Main processing loop
    remaining_jobs = total_jobs - start_index
    
    # Always use batch size for logical grouping, but process sequentially
    current_batch_size = BATCH_SIZE
    
    print_console(f"🎯 Processing {remaining_jobs} jobs (sequential calls, logical batch size: {current_batch_size})...")
    
    # Calculate number of batches
    num_batches = math.ceil(remaining_jobs / current_batch_size)
    
    # Setup tqdm progress bar for batches
    batch_progress_bar = tqdm(
        total=num_batches,
        desc="🔄 Processing",
        unit="batch",
        position=0,
        leave=True,
        bar_format='{l_bar}{bar:20}{r_bar}'
    )
    
    for batch_num in range(num_batches):
        batch_start = start_index + (batch_num * current_batch_size)
        batch_end = min(batch_start + current_batch_size, total_jobs)
        current_batch = shortlisted[batch_start:batch_end]
        
        # Update progress bar description
        batch_progress_bar.set_description(f"🔄 Batch {batch_num+1}/{num_batches}")
        
        try:
            # Process batch sequentially (no generate_batch)
            batch_results = process_job_batch(
                gemini_client, 
                current_batch, 
                RESUME_DIR,
                batch_idx=batch_num,
                total_batches=num_batches
            )
            
            # Check if processing was stopped due to exhausted API keys
            if batch_results and any(_is_no_key_available(r) for r in batch_results):
                print_console(f"\n⚠️ All API keys exhausted or cooling down — stopping entire process")
                # Save progress for the jobs processed so far
                save_progress(batch_start, total_jobs, results + batch_results)
                batch_progress_bar.update(1)
                break
            
            # Save batch results to database
            successful_in_batch = 0
            for result in batch_results:
                if save_to_db(db_conn, result):
                    results.append(result)
                    if result["status"] == "success":
                        successful_in_batch += 1
            
            # Update progress
            current_index = batch_end - 1
            save_progress(current_index, total_jobs, results)
            
            # Update progress bar
            batch_progress_bar.update(1)
            batch_progress_bar.set_postfix({
                'success': f"{successful_in_batch}/{len(current_batch)}",
                'total': f"{len(results)}/{total_jobs}"
            })
            
            # Rate limiting between batches
            if batch_num < num_batches - 1:
                time.sleep(1)
            
        except KeyboardInterrupt:
            print_console(f"\n⏸️ Process interrupted")
            save_progress(batch_start, total_jobs, results)
            db_conn.close()
            batch_progress_bar.close()
            return
            
        except Exception as e:
            logging.exception(f"Batch {batch_num + 1} failed: {e}")
            print(f"❌ Batch {batch_num + 1} failed with error: {e}")
            batch_progress_bar.update(1)
            continue

    # Close progress bar
    batch_progress_bar.close()
    
    # Final cleanup and summary
    db_conn.close()
    
    # Save final JSON export
    try:
        with open(RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to save JSON export: {e}")

    # Calculate statistics
    elapsed_time = time.time() - start_time
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    total_tokens = sum(r.get("total_estimated_tokens", 0) for r in results)
    
    # CONSOLE: Clean final summary
    print_console("\n" + "=" * 40)
    print_console("🎉 ANALYSIS COMPLETED!")
    print_console("=" * 40)
    
    print_summary("📊 Results:", [
        f"✅ Successful: {successful} jobs",
        f"❌ Failed: {failed} jobs", 
        f"📈 Success Rate: {successful/len(results)*100:.1f}%" if results else "N/A"
    ])
    
    print_summary("⏱️  Performance:", [
        f"Total Time: {elapsed_time/60:.1f} minutes",
        f"Jobs/Minute: {len(results)/elapsed_time*60:.1f}" if elapsed_time > 0 else "N/A",
        f"Total Tokens: {total_tokens:,}",
        f"Estimated Cost: ${total_tokens/10000 * 0.375:.4f}"
    ])
    
    print_summary("💾 Output Files:", [
        f"Database: {RESULT_DB}",
        f"JSON Export: {RESULT_JSON}",
        f"Detailed Log: {log_file}"
    ])
    
    # Print Gemini stats to console
    if hasattr(gemini_client, 'print_stats'):
        gemini_client.print_stats()
    
    cleanup_progress()
    print_console("\n🏁 Done!")

if __name__ == "__main__":
    main()
