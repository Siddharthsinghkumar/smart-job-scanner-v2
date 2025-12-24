#!/usr/bin/env python3
"""
12_auto_download_pdfs.py – Complete newspaper downloader with Telegram support
"""

import requests
import os
import time
import json
import asyncio
import urllib.parse
import tempfile
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
import logging
from logging.handlers import RotatingFileHandler
import concurrent.futures
from threading import RLock
import queue
from urllib.parse import urlparse, parse_qs, urljoin
import re
from difflib import SequenceMatcher
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Silence Selenium and other noisy modules
os.environ["WDM_LOG_LEVEL"] = "0"
os.environ["NO_COLOR"] = "1"

for noisy_mod in ["selenium", "telethon", "urllib3", "asyncio", "charset_normalizer"]:
    logging.getLogger(noisy_mod).setLevel(logging.ERROR)

# Optional Telegram imports
try:
    from telethon import TelegramClient
    from telethon.tl.types import MessageMediaDocument
except ImportError:
    TelegramClient = None
    MessageMediaDocument = None

# Filename sanitization regex
_filename_sanitize_re = re.compile(r'[^\w\-. ]')

def sanitize_filename(name: str, max_len: int = 200) -> str:
    """Make a safe filename from an arbitrary string."""
    if not name:
        return "file"
    s = str(name)
    s = s.strip().replace(" ", "_")
    s = _filename_sanitize_re.sub("_", s)
    return s[:max_len]

def atomic_write_stream(resp, out_path: Path, chunk_size: int = 8192):
    """Write stream to a temporary file and atomically replace target."""
    download_dir = out_path.parent
    download_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(download_dir), suffix=".part") as tmp:
        tmp_name = Path(tmp.name)
        try:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    tmp.write(chunk)
            tmp.flush()
            os.fsync(tmp.fileno())
            # atomic replace
            os.replace(str(tmp_name), str(out_path))
            return True
        except Exception:
            # cleanup partial
            try:
                tmp_name.unlink(missing_ok=True)
            except Exception:
                pass
            raise

def normalize_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[\-\_\(\)\[\]\.]', ' ', text)   # remove punctuation
    text = re.sub(r'\d{1,2}[-/]\d{1,2}|\d{4}', '', text)  # remove dates
    text = re.sub(r'\s+', ' ', text).strip()  # collapse spaces
    return text

def fuzzy_match(name, candidate):
    name = normalize_text(name)
    cand = normalize_text(candidate)

    # direct contains
    if name in cand or cand in name:
        return True
    
    # fuzzy ratio
    ratio = SequenceMatcher(None, name, cand).ratio()
    return ratio >= 0.65  # threshold

def clean_name(name):
    s = normalize_text(name)
    s = s.replace("express", "ie")
    return s

def status(msg):
    """Single-line status updates for clean terminal output"""
    print(f"\r{msg:<80}", end="", flush=True)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # master logger

# File handler (rotating logs)
log_path = Path("newspaper_downloader.log")
fh = RotatingFileHandler(str(log_path), maxBytes=5 * 1024 * 1024, backupCount=5)
fh.setLevel(logging.DEBUG)   # keep everything in file
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Console handler (reduced verbosity)
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)  # <--- only show WARN+ on terminal (info stays in file)
ch.setFormatter(logging.Formatter('%(message)s'))

# Avoid duplicate handlers (compare by class & formatter)
existing = {(type(h), getattr(h, 'formatter', None).__class__) for h in logger.handlers}
if (type(fh), fh.formatter.__class__) not in existing:
    logger.addHandler(fh)
if (type(ch), ch.formatter.__class__) not in existing:
    logger.addHandler(ch)

class CompleteNewspaperDownloader:
    def __init__(self, config_file="newspaper_config.json", batch_size=4, max_preprocess_workers=4):
        self.config = self.load_config(config_file)
        self.batch_size = batch_size
        self.max_preprocess_workers = max_preprocess_workers
        
        # Configurable thresholds
        self.MAX_CANDIDATES = self.config["settings"].get("max_candidates", 6)
        self.MAX_DOWNLOAD_DEPTH = self.config["settings"].get("max_download_depth", 5)
        self.MIN_ACCEPTED_FILESIZE = int(self.config["settings"].get("min_filesize_bytes", 100000))
        
        # Create session with retry mechanism
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.session.headers.update({
            "User-Agent": self.config["settings"]["user_agent"]
        })
        
        # how long to pause after first navigation to allow JS/cloudflare checks (seconds)
        self.INITIAL_PAGE_WAIT = int(self.config["settings"].get("initial_page_wait", 10))
        
        self.download_lock = RLock()  # Changed from Lock to RLock
        self.already_downloaded = self.check_existing_downloads()
        
        # Track files downloaded in this run
        self.run_downloaded_files = []
        
        # Backward compatibility: support both gdrive_sources and web_sources
        if "web_sources" in self.config:
            self.web_sources = self.config["web_sources"]
            logger.info("✅ Using web_sources from config")
        elif "gdrive_sources" in self.config:
            self.web_sources = self.config["gdrive_sources"]
            logger.info("🔄 Using gdrive_sources (backward compatibility)")
        else:
            self.web_sources = {}
            logger.warning("⚠️ No web sources found in config")
        
        # For tracking concurrent downloads of the same URL
        self._in_progress = set()
        
        self.setup_selenium_pool()
        
    # ---------- State helpers for skip / retry ----------
    def _state_file(self):
        return Path(self.config["settings"]["download_dir"]) / ".download_state.json"

    def load_state(self):
        sf = self._state_file()
        try:
            if sf.exists():
                return json.loads(sf.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Could not load state file, starting fresh")
        return {}

    def save_state(self, state):
        sf = self._state_file()
        try:
            sf.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug(f"Could not write state file: {e}")

    def mark_skip_gdrive(self, file_id, reason="scan_protect", hours=24):
        """Mark a GDrive file_id as skipped until now + hours"""
        state = self.load_state()
        skips = state.get("gdrive_skips", {})
        expiry_ts = (datetime.now() + timedelta(hours=hours)).timestamp()
        skips[file_id] = {"expiry": expiry_ts, "reason": reason, "marked_at": datetime.now().isoformat()}
        state["gdrive_skips"] = skips
        self.save_state(state)
        logger.info(f"⏳ Marked GDrive {file_id} skipped for {hours}h because: {reason}")

    def should_skip_gdrive(self, file_id):
        state = self.load_state()
        skips = state.get("gdrive_skips", {})
        if file_id in skips:
            try:
                if datetime.now().timestamp() < float(skips[file_id]["expiry"]):
                    return True
                else:
                    # expired; remove
                    del skips[file_id]
                    state["gdrive_skips"] = skips
                    self.save_state(state)
            except Exception:
                return False
        return False

    def validate_pdf_file(self, path: Path):
        """Return True if file looks like a valid PDF (simple header + size check)."""
        try:
            if not path.exists():
                return False
            size = path.stat().st_size
            if size < self.MIN_ACCEPTED_FILESIZE:
                logger.warning(f"⚠️ PDF too small: {path} ({size} bytes)")
                return False
            with open(path, "rb") as f:
                header = f.read(5)
                if header.startswith(b"%PDF"):
                    # Read the last 1KB to check for EOF
                    try:
                        f.seek(max(0, size - 1024))
                        tail = f.read(1024)
                        if b"%%EOF" in tail:
                            return True
                    except Exception:
                        # If we can't read the tail, still accept the header
                        return True
                return False
        except Exception as e:
            logger.debug(f"validate_pdf_file error: {e}")
            return False
    
    def finalize_download(self, out_file: Path, source_name: str):
        """Validate a downloaded file, update bookkeeping, and return success."""
        try:
            if not out_file.exists() or out_file.stat().st_size <= self.MIN_ACCEPTED_FILESIZE:
                logger.warning(f"⚠️ Downloaded file missing or too small: {out_file}")
                try:
                    out_file.unlink(missing_ok=True)
                except Exception:
                    pass
                return False
            if not self.validate_pdf_file(out_file):
                logger.warning(f"⚠️ Downloaded file failed PDF validation: {out_file}")
                try:
                    out_file.unlink(missing_ok=True)
                except Exception:
                    pass
                return False
            # success: update sets
            self.already_downloaded.add(source_name)
            self.run_downloaded_files.append(out_file.name)
            logger.info(f"✅ Finalized download: {out_file.name}")
            return True
        except Exception as e:
            logger.error(f"❌ finalize_download error: {e}")
            return False
        
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            download_dir = Path(config["settings"]["download_dir"])
            download_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"✅ Configuration loaded from {config_file}")
            return config
        except Exception as e:
            logger.error(f"❌ Error loading config file: {e}")
            raise

    def check_existing_downloads(self):
        """Check which files have already been downloaded today"""
        download_dir = Path(self.config["settings"]["download_dir"])
        today_str = datetime.now().strftime("%Y%m%d")
        
        existing_files = set()
        
        try:
            for file_path in download_dir.glob(f"*_{today_str}.pdf"):
                source_name = file_path.stem.replace(f"_{today_str}", "")
                if file_path.stat().st_size > self.MIN_ACCEPTED_FILESIZE and self.validate_pdf_file(file_path):
                    existing_files.add(source_name)
                    status(f"✅ Already downloaded {source_name}, skipping...")
                    logger.info(f"📁 Already downloaded: {source_name} ({file_path.stat().st_size} bytes)")
                else:
                    logger.warning(f"⚠️ Invalid or small file detected, may re-download: {source_name}")
                    try:
                        file_path.unlink(missing_ok=True)
                        logger.info(f"🗑️  Deleted invalid file: {source_name}")
                    except:
                        pass
        
        except Exception as e:
            logger.error(f"❌ Error checking existing downloads: {e}")
        
        logger.info(f"📊 Found {len(existing_files)} already downloaded files for today")
        return existing_files

    def _move_latest_browser_download(self, source_name: str, wait_seconds: int = 20, poll_interval: float = 0.5):
        """
        After Selenium triggers a browser download, Chrome will write a file (possibly with a random name)
        into the configured download dir. Poll the download dir for a new file and move/rename it to the
        canonical sanitized filename. Returns Path on success, or None.
        """
        download_dir = Path(self.config["settings"]["download_dir"]).resolve()
        if not download_dir.exists():
            return None

        # snapshot before wait
        before = {p.name: p.stat().st_mtime for p in download_dir.glob("*") if p.is_file()}

        deadline = time.time() + wait_seconds
        found = None
        while time.time() < deadline:
            # look for files with recent mtime or temp extensions
            for p in download_dir.glob("*"):
                if not p.is_file():
                    continue
                if p.name in before:
                    # maybe it was overwritten; check mtime change
                    if p.stat().st_mtime > before[p.name]:
                        found = p
                        break
                    else:
                        continue
                # new file added
                # skip partial download files that Chrome leaves while still writing (.crdownload)
                if p.suffix.lower() in ['.crdownload', '.part', '.tmp']:
                    continue
                # also accept files that look like random hex names (no underscore + date)
                found = p
                break
            if found:
                break
            time.sleep(poll_interval)

        if not found:
            # final defensive scan for completed .crdownload -> renamed file
            for p in download_dir.glob("*"):
                if p.suffix.lower() in ['.crdownload', '.part', '.tmp']:
                    # if there's an accompanying completed file without extension, try to pick it up next loop
                    continue

        if not found:
            return None

        # rename/move to canonical safe name
        today_str = datetime.now().strftime("%Y%m%d")
        safe_name = sanitize_filename(f"{source_name}_{today_str}.pdf")
        target = download_dir / safe_name

        try:
            # If target exists, append a suffix to avoid clobber
            if target.exists():
                target = download_dir / sanitize_filename(f"{source_name}_{today_str}_{int(time.time())}.pdf")
            found.rename(target)
            logger.info(f"🔁 Moved browser-downloaded file {found.name} -> {target.name}")
            return target
        except Exception as e:
            logger.debug(f"Could not rename/move browser download {found} -> {target}: {e}")
            # try shutil.move as fallback
            try:
                shutil.move(str(found), str(target))
                return target
            except Exception as ex:
                logger.error(f"❌ Failed to move browser download: {ex}")
                return None

    def wait_for_js_checks(self, driver, timeout=30, check_interval=1):
        """Wait until the page is no longer the Cloudflare 'checking your browser' interstitial."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                src = (driver.page_source or "").lower()
                if ('checking your browser' not in src and 
                    'please wait' not in src and 
                    'var _cf_chl_ctx' not in src and
                    'ddos' not in src and
                    'just a moment' not in src):
                    return True
            except Exception:
                pass
            time.sleep(check_interval)
        return False

    def setup_selenium_pool(self):
        """Setup a pool of Selenium drivers for parallel preprocessing"""
        self._closing = False
        self.driver_pool = queue.Queue()
        
        # Check if we have sources that need preprocessing
        sources_to_process = [
            name for name in self.web_sources.keys() 
            if name not in self.already_downloaded
        ]
        
        if not sources_to_process:
            logger.info("ℹ️ No web sources need preprocessing - all already downloaded")
            return
        
        chrome_options = Options()
        chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])  # Suppress DevTools logs
        
        # Anti-detection fingerprint hiding
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        
        if self.config["settings"].get("headless", True):
            chrome_options.add_argument("--headless=new")  # try new headless mode
        else:
            # headful (visible) — useful for tricky Drive 'download anyway' flows
            pass
        
        # Set Chrome prefs + enable downloads in headless via CDP
        download_dir = str(Path(self.config["settings"]["download_dir"]).resolve())
        prefs = {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True,  # don't open PDFs in Chrome's viewer
            "profile.default_content_settings.popups": 0,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")  # sometimes helps renderer crashes
        
        num_drivers_needed = min(self.max_preprocess_workers, len(sources_to_process))
        for i in range(num_drivers_needed):
            try:
                # Create driver with CDP download behavior
                driver = webdriver.Chrome(options=chrome_options)
                
                # Anti-detection: Hide webdriver property
                try:
                    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
                    })
                except Exception as e:
                    logger.debug(f"Could not apply webdriver hide script: {e}")
                
                # IMPORTANT: for headless Chrome, enable download behavior via CDP
                try:
                    # Chrome >= 75 supports this CDP call; works for headless & headful
                    driver.execute_cdp_cmd("Page.setDownloadBehavior", {
                        "behavior": "allow",
                        "downloadPath": download_dir
                    })
                except Exception as e:
                    logger.debug(f"Could not set CDP download behavior: {e}")
                driver.set_page_load_timeout(120)
                self.driver_pool.put(driver)
                logger.info(f"✅ Created Selenium driver {i+1}/{num_drivers_needed} with download_dir={download_dir}")
            except Exception as e:
                logger.error(f"❌ Failed to create Selenium driver {i+1}: {e}")

    def get_driver(self, wait_timeout=0):
        """Get a driver from the pool. If wait_timeout>0, block up to that many seconds."""
        if self._closing:
            return None
        try:
            if wait_timeout and wait_timeout > 0:
                return self.driver_pool.get(timeout=wait_timeout)
            else:
                if self.driver_pool.empty():
                    return None
                return self.driver_pool.get_nowait()
        except queue.Empty:
            return None

    def return_driver(self, driver):
        """Return a driver to the pool"""
        if driver and not self._closing:
            self.driver_pool.put(driver)

    def close_drivers(self):
        """Close all drivers in the pool"""
        self._closing = True
        if hasattr(self, 'driver_pool'):
            logger.info("🔄 Closing Selenium drivers...")
            while not self.driver_pool.empty():
                try:
                    driver = self.driver_pool.get_nowait()
                    driver.quit()
                except:
                    pass
            logger.info("✅ All Selenium drivers closed")

    def dismiss_overlays(self, driver):
        """Best-effort: remove modal/overlay/popups and try clicking 'close' buttons."""
        try:
            # quick JS removal of common overlay/modal selectors
            js = r"""
            try {
                var selectors = [
                    'div[class*="modal"]','div[class*="overlay"]','div[class*="popup"]',
                    'div[class*="subscribe"]','div[class*="newsletter"]','div[class*="tg-widget"]',
                    'iframe[src*="telegram"]','div[class*="telegram"]','div[class*="subscribe"]'
                ];
                selectors.forEach(function(s){
                    document.querySelectorAll(s).forEach(function(e){ e.remove(); });
                });
                // also try removing elements fixed to viewport
                document.querySelectorAll('body > *').forEach(function(e){
                    var st = window.getComputedStyle(e);
                    if(st && (st.position === 'fixed' || st.position === 'sticky')) { try { e.remove(); } catch(e){} }
                });
            } catch(e){}
            """
            driver.execute_script(js)
        except Exception as e:
            logger.debug(f"overlay removal JS failed: {e}")

        # try clicking buttons that look like 'close' / 'dismiss'
        try:
            close_buttons = driver.find_elements(By.XPATH,
                "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'close') or contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'dismiss') or contains(., '×') or contains(., 'cancel') or contains(., 'no thanks')]")
            for b in close_buttons:
                try:
                    b.click()
                    time.sleep(0.3)
                except:
                    pass
        except Exception:
            pass
        
        # Additional aggressive fallback: make fixed elements non-interactive
        try:
            driver.execute_script("""
                document.querySelectorAll('div, iframe').forEach(function(e){
                    if(window.getComputedStyle(e).position === 'fixed' || e.style.zIndex > 1000){
                        e.style.pointerEvents='none';
                        e.style.opacity='0.01';
                    }
                });
            """)
        except Exception as e:
            logger.debug(f"pointer-events fallback failed: {e}")

    def save_debug_page(self, source_name, html, note=""):
        """Save debug HTML for troubleshooting"""
        debug_dir = Path(self.config["settings"]["download_dir"]) / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        note_safe = f"_{note}" if note else ""
        filename = debug_dir / sanitize_filename(f"{source_name}_{timestamp}{note_safe}.html")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"📄 Saved debug page for {source_name} to {filename}")

    def get_todays_date_formats(self, date_format=None, as_list=False):
        """Get today's date in multiple formats"""
        today = datetime.now()
        
        # If date_format is provided, use it as primary
        formats = []
        if date_format:
            if isinstance(date_format, list):
                # If config provides multiple formats
                for fmt in date_format:
                    formats.append(today.strftime(fmt))
            else:
                formats.append(today.strftime(date_format))
        
        # Always add common date formats
        formats.extend([
            today.strftime("%d-%m-%Y"),      # 11-12-2024
            today.strftime("%d/%m/%Y"),      # 11/12/2024
            today.strftime("%d %B %Y"),      # 11 December 2024
            today.strftime("%d %b %Y"),      # 11 Dec 2024
            today.strftime("%d-%m-%y"),      # 11-12-24
            today.strftime("%d/%m/%y"),      # 11/12/24
            today.strftime("%d %B, %Y"),     # 11 December, 2024
            today.strftime("%d %b, %Y"),     # 11 Dec, 2024
            today.strftime("%Y-%m-%d"),      # 2024-12-11
            today.strftime("%Y%m%d"),        # 20241211
            today.strftime("%d%m%Y"),        # 11122024
            today.strftime("%d-%b-%Y"),      # 11-Dec-2024
            today.strftime("%d.%m.%Y"),      # 11.12.2024
            today.strftime("%B %d, %Y"),     # December 14, 2025
            today.strftime("%b %d, %Y"),     # Dec 14, 2025
            today.strftime("%B %d"),         # December 14
            today.strftime("%b %d"),         # Dec 14
            today.strftime("%d %B"),         # 14 December
            today.strftime("%d %b"),         # 14 Dec
            today.strftime("%d %B, %Y"),     # 14 December, 2025
        ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_formats = []
        for fmt in formats:
            if fmt not in seen:
                seen.add(fmt)
                unique_formats.append(fmt)
        
        if as_list:
            return unique_formats
        else:
            # For backward compatibility, return the first format
            return unique_formats[0] if unique_formats else today.strftime("%d-%m-%Y")

    def extract_gdrive_file_id(self, url):
        """Extract Google Drive file ID from various URL formats (drive.google.com, usercontent, googleusercontent)."""
        try:
            if not url:
                return None
            url = str(url)
            # common patterns
            m = re.search(r"/file/d/([a-zA-Z0-9_-]{10,})", url)
            if m:
                return m.group(1)
            m = re.search(r"[?&]id=([a-zA-Z0-9_-]{10,})", url)
            if m:
                return m.group(1)
            # usercontent / googleusercontent sometimes embed ids like /d/<id>/ or /uc?id=<id>
            m = re.search(r"/d/([a-zA-Z0-9_-]{10,})", url)
            if m:
                return m.group(1)
            m = re.search(r"/uc\?id=([a-zA-Z0-9_-]{10,})", url)
            if m:
                return m.group(1)
            # fallback: parse query params
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            if 'id' in params:
                return params['id'][0]
            return None
        except Exception as e:
            logger.debug(f"Error extracting gdrive id: {e}")
            return None

    def extract_gdrive_resource_key(self, url):
        """Extract Drive resourcekey (if present) from URL query params."""
        try:
            if not url:
                return None
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            # common param name is 'resourcekey' (case-insensitive)
            for k, v in qs.items():
                if k.lower() == "resourcekey" and v:
                    return v[0]
            return None
        except Exception as e:
            logger.debug(f"Error extracting resourcekey: {e}")
            return None

    def try_resolve_viewer_link(self, text_or_url):
        """Given text or a URL, try to find Drive/docs viewer/pdf and return a direct PDF URL if possible."""
        # look for direct pdf urls
        m = re.search(r"https?://[^\s'\"<>]+\.pdf", text_or_url, flags=re.I)
        if m:
            return m.group(0)

        # google drive file id patterns
        m = re.search(r"/file/d/([a-zA-Z0-9_-]{10,})", text_or_url)
        if m:
            fid = m.group(1)
            return f"https://drive.google.com/uc?id={fid}&export=download"

        m = re.search(r"[?&]id=([a-zA-Z0-9_-]{10,})", text_or_url)
        if m:
            fid = m.group(1)
            return f"https://drive.google.com/uc?id={fid}&export=download"

        # docs viewer patterns that embed a URL param
        m = re.search(r"(?:viewer|gview)[^\s'\"<>]*\?url=(https?%3A%2F%2F[^\s'\"<>]+)", text_or_url)
        if m:
            # url may be percent-encoded
            try:
                decoded = urllib.parse.unquote(m.group(1))
                if decoded.lower().endswith('.pdf'):
                    return decoded
            except Exception:
                pass

        # docs.google.com/viewer?url=... or other embed styles
        m = re.search(r"https?://docs\.google\.com/[^\s'\"<>]+", text_or_url)
        if m:
            # often viewer links point to 3rd-party pdf-hosted urls; try to extract .pdf from it
            inner = m.group(0)
            m2 = re.search(r"https?://[^\s'\"<>]+\.pdf", inner)
            if m2:
                return m2.group(0)

        return None

    def detect_link_type(self, url):
        """Automatically detect whether a URL is Google Drive, direct PDF, or other"""
        if not url:
            return "unknown"
        s = str(url).lower()
        # Check for Google Drive domains and patterns
        if any(domain in s for domain in ['drive.google.com', 'drive.googleusercontent.com', 'drive.usercontent.google.com']) or '/uc?id=' in s:
            return "gdrive"
        elif s.endswith('.pdf'):
            return "direct_pdf"
        elif 'pdf' in s:
            return "likely_pdf"
        else:
            return "unknown"

    def copy_selenium_cookies_to_session(self, driver, sess):
        """Copy cookies from Selenium driver to requests session correctly."""
        try:
            for c in driver.get_cookies():
                name = c.get("name")
                value = c.get("value", "")
                domain = c.get("domain", None)
                path = c.get("path", "/")
                secure = c.get("secure", False)
                http_only = c.get("httpOnly", False)

                # Requests requires domain without leading dot sometimes; accept both
                if domain:
                    try:
                        sess.cookies.set(name, value, domain=domain, path=path)
                    except Exception:
                        # fallback: set without domain
                        sess.cookies.set(name, value, path=path)
                else:
                    sess.cookies.set(name, value, path=path)
        except Exception as e:
            logger.debug(f"Could not copy selenium cookies: {e}")

    def preprocess_epaperwave(self, source_name, source_config):
        """Specialized handler for epaperwave.com websites"""
        url = source_config["url"]
        driver = self.get_driver(wait_timeout=10)
        
        if not driver:
            return {"source": source_name, "success": False, "error": "No driver available"}
            
        try:
            # Use try/except for TimeoutException
            try:
                driver.get(url)
                # Log the final URL
                try:
                    cur = driver.current_url
                    logger.info(f"🔗 Selenium ended at URL: {cur}")
                except Exception:
                    pass
                # Wait a bit for initial load
                time.sleep(1)
                # Wait for Cloudflare/JS checks to complete (configurable)
                logger.debug(f"⏱️ Waiting {self.INITIAL_PAGE_WAIT}s for JS/cloudflare to finish on {url}")
                time.sleep(self.INITIAL_PAGE_WAIT)
                # Then actively wait for cloudflare to finish (up to 30s)
                if not self.wait_for_js_checks(driver, timeout=30):
                    logger.debug("⏱️ Cloudflare/JS check still present after timeout")
            except TimeoutException:
                logger.warning(f"⚠️ driver.get() timed out for {url} — continuing with partial DOM")
            
            # --- PATCH 3: Abort if redirected to Telegram ---
            cur = driver.current_url.lower()
            if "t.me" in cur or "telegram" in cur:
                logger.info(f"❌ Aborted: Page redirected to Telegram → {cur}")
                return {"source": source_name, "success": False, "error": "Redirected to Telegram"}
            
            # Dismiss overlays before processing
            self.dismiss_overlays(driver)
            
            # quick check: if the driver redirected to a Telegram page, skip this candidate
            try:
                cur = driver.current_url.lower()
                if 't.me' in cur or 'telegram' in cur:
                    logger.info(f"🔕 Page redirected to Telegram ({cur}) — skipping")
                    return {"source": source_name, "success": False, "error": "Page redirected to Telegram"}
            except Exception:
                pass
            
            # give a small pause to let any JS render
            time.sleep(2)
            
            # Try to click common cookie accept buttons (best-effort)
            try:
                btn = driver.find_element(By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept') or contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'agree') or contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'close')]")
                btn.click()
                time.sleep(1)
            except Exception:
                pass
                
            soup = BeautifulSoup(driver.page_source, "html.parser")

            # Robust date detection
            today = datetime.now()
            day = today.day
            month_long = today.strftime("%B")
            month_short = today.strftime("%b")
            year = today.year

            # patterns: numeric and textual
            numeric_pat = re.compile(rf"\b{day}\s*[-/]\s*{today.strftime('%m')}\s*[-/]\s*{year}\b")
            numeric_pat2 = re.compile(rf"\b{day}\s*[-/]\s*{today.strftime('%m')}\s*[-/]\s*{str(year)[-2:]}\b")
            text_pat = re.compile(rf"\b{day}(?:st|nd|rd|th)?\b.*?\b{month_long}\b.*?{year}", re.I)
            month_section_pat = re.compile(rf"{re.escape(month_long)}\s*{year}", re.I)

            found_block = None

            # 1) try exact numeric variants
            for pat in [numeric_pat, numeric_pat2]:
                candidate = soup.find(string=pat)
                if candidate:
                    found_block = candidate.parent
                    break

            # 2) try textual day+month pattern
            if not found_block:
                candidate = soup.find(string=text_pat)
                if candidate:
                    found_block = candidate.parent

            # 3) if still not found, look for a month-year header then find a nearby list item with the day number
            if not found_block:
                month_header = soup.find(string=month_section_pat)
                if month_header:
                    # look for nearest container that has day links / anchors
                    parent = month_header.parent
                    found_li = None
                    # walk up a couple levels then search down
                    for _ in range(4):
                        if parent is None:
                            break
                        # find anchors or list items that contain standalone day number
                        candidates = parent.find_all(lambda tag: tag.name in ('a','li','span','div') and re.search(rf"\b{day}\b", (tag.get_text() or "")))
                        if candidates:
                            found_li = candidates[0]
                            break
                        parent = parent.parent
                    if found_li:
                        found_block = found_li

            if not found_block:
                # Fallback: scanning whole page for edition/download links for {source_name}")
                anchors = soup.find_all('a', href=True)
                for a in anchors:
                    txt = (a.get_text() or "").lower()
                    href = a['href']
                    if any(k in txt for k in ('edition','editions','download','epaper','view','pdf')) or '.pdf' in href.lower():
                        resolved = urljoin(url, href)
                        logger.info(f"🔎 Fallback pick -> {resolved} (anchor text: {txt[:80]})")
                        link_type = self.detect_link_type(resolved)
                        if link_type == "gdrive":
                            file_id = self.extract_gdrive_file_id(resolved)
                            if file_id:
                                return {"source": source_name, "success": True, "file_id": file_id, "url": resolved, "type": "gdrive", "already_downloaded": False}
                            else:
                                return {"source": source_name, "success": True, "url": resolved, "type": "direct_pdf", "already_downloaded": False}
                        else:
                            return {"source": source_name, "success": True, "url": resolved, "type": "direct_pdf", "already_downloaded": False}
                # still not found -> save debug and return
                self.save_debug_page(source_name, driver.page_source, note="no_date_no_fallback")
                return {"source": source_name, "success": False, "error": "Date not found and no fallback link"}

            # Collect candidate anchors from siblings + container, dedupe and prioritize
            links = []
            node = found_block
            for _ in range(10):
                node = node.find_next_sibling() if node is not None else None
                if not node:
                    break
                links += list(node.find_all("a", href=True))

            container = found_block.find_parent()
            if container:
                links += list(container.find_all("a", href=True))

            # Deduplicate & prioritize
            seen = set()
            candidates = []
            
            # Prepare newspaper name variants
            variants = [source_name.replace('_', ' ')]
            # allow user-defined variants in the config (optional)
            for v in source_config.get("variants", []):
                variants.append(v)
            
            # First pass: find anchors that match newspaper name
            matched_by_name = False
            for a in links:
                text = (a.get_text(strip=True) or "").lower()
                href = a.get("href", "").strip()
                if not href or href.startswith("javascript:") or href.startswith("mailto:"):
                    continue
                
                # Skip Telegram links
                if 't.me' in href or 'telegram' in href:
                    logger.info(f"🔕 Skipping telegram link while preprocessing: {href}")
                    continue
                    
                resolved = urljoin(url, href)
                key = (resolved,)
                if key in seen:
                    continue
                seen.add(key)
                
                # if anchor text fuzzily matches the newspaper name, push it to front
                if any(fuzzy_match(v, text) for v in variants):
                    if (resolved.lower().endswith(".pdf") or ".pdf" in resolved.lower() or "/download" in resolved.lower()):
                        candidates.insert(0, (a, resolved))
                    else:
                        candidates.insert(0, (a, resolved))
                    matched_by_name = True
            
            # If nothing matched by fuzzy rules, fall back to original heuristics
            if not matched_by_name:
                for a in links:
                    href = a.get("href", "").strip()
                    if not href or href.startswith("javascript:") or href.startswith("mailto:"):
                        continue
                    
                    # Skip Telegram links
                    if 't.me' in href or 'telegram' in href:
                        logger.info(f"🔕 Skipping telegram link while preprocessing: {href}")
                        continue
                    
                    resolved = urljoin(url, href)
                    key = (resolved,)
                    if key in seen:
                        continue
                    seen.add(key)
                    text = (a.get_text(strip=True) or "").lower()
                    if resolved.lower().endswith(".pdf") or ".pdf" in resolved.lower() or "/download" in resolved.lower():
                        candidates.insert(0, (a, resolved))
                    elif any(k in text for k in ['delhi','edition','editions','other editions','download','epaper','pdf','view']):
                        candidates.append((a, resolved))

            # Aggressive attribute + JS-string PDF detection
            # collect extra candidates from data-* attrs, onclicks, and scripts
            extra_candidates = set()
            for tag in soup.find_all(True):
                for attr in ('data-href','data-src','data','data-original','data-url','onclick','href','src'):
                    if tag.has_attr(attr):
                        val = (tag.get(attr) or "").strip()
                        if val:
                            # possibly contains encoded or direct pdf/drive url
                            if '.pdf' in val.lower() or 'drive.google.com' in val.lower() or 'uc?export=download' in val.lower():
                                extra_candidates.add(urljoin(url, val))
            # Check inline scripts
            for script in soup.find_all("script"):
                st = script.string or ""
                if st and ('.pdf' in st.lower() or 'drive.google.com' in st.lower()):
                    m = re.search(r"https?://[^\s'\"<>]+\.pdf", st, flags=re.I)
                    if m:
                        extra_candidates.add(m.group(0))
                    # also look for drive patterns
                    m2 = re.search(r"/uc\?id=[A-Za-z0-9_\-]+", st)
                    if m2:
                        extra_candidates.add(urljoin("https://drive.google.com", m2.group(0)))

            # append these candidates to your regular 'candidates' list (dedupe)
            for cand in extra_candidates:
                candidates.append((None, cand))

            if not candidates:
                # fallback: include some anchors but limit count
                for a in links:
                    href = a.get("href", "").strip()
                    if href and href.startswith("http"):
                        # Skip Telegram links
                        if 't.me' in href or 'telegram' in href:
                            continue
                        candidates.append((a, urljoin(url, href)))
                        if len(candidates) >= 8:
                            break

            # Follow candidate edition pages and aggressively search for PDFs
            for idx, (a, full) in enumerate(candidates):
                if idx >= self.MAX_CANDIDATES:
                    logger.info(f"⏭️ Reached candidate limit ({self.MAX_CANDIDATES}) for {source_name}, stopping candidate scan")
                    break
                    
                try:
                    # Use try/except for TimeoutException
                    try:
                        driver.get(full)
                        # Log the final URL
                        try:
                            cur = driver.current_url
                            logger.info(f"🔗 Selenium ended at URL: {cur}")
                        except Exception:
                            pass
                        # WAIT on first candidate navigation to allow checks to finish
                        if idx == 0:
                            logger.debug(f"⏱️ Waiting {self.INITIAL_PAGE_WAIT}s after opening first candidate {full}")
                            time.sleep(self.INITIAL_PAGE_WAIT)
                            # Then actively wait for cloudflare to finish
                            if not self.wait_for_js_checks(driver, timeout=30):
                                logger.debug("⏱️ Cloudflare/JS check still present after timeout")
                    except TimeoutException:
                        logger.warning(f"⚠️ driver.get() timed out for {full} — continuing with partial DOM")
                    
                    # quick check: if the driver redirected to a Telegram page, skip this candidate
                    try:
                        cur = driver.current_url.lower()
                        if 't.me' in cur or 'telegram' in cur:
                            logger.info(f"🔕 candidate redirected to Telegram ({cur}) — skipping")
                            continue  # skip this candidate
                    except Exception:
                        pass
                    
                    # Dismiss overlays before processing
                    self.dismiss_overlays(driver)
                    
                    # --- Check for iframes that point to Google Drive viewers and use that src directly ---
                    try:
                        frames = driver.find_elements(By.TAG_NAME, "iframe")
                        for fr in frames:
                            try:
                                src = (fr.get_attribute("src") or fr.get_attribute("data-src") or "").strip()
                                if src:
                                    resolved_src = urljoin(full, src)
                                    fid = self.extract_gdrive_file_id(resolved_src)
                                    if fid:
                                        logger.info(f"🔎 Found drive iframe src; routing to drive handler: {resolved_src}")
                                        return {"source": source_name, "success": True, "file_id": fid, "url": resolved_src, "type": "gdrive", "already_downloaded": False}
                                    elif resolved_src.lower().endswith('.pdf') or '.pdf' in resolved_src.lower():
                                        return {"source": source_name, "success": True, "url": resolved_src, "type": "direct_pdf", "already_downloaded": False}
                            except Exception:
                                continue
                    except Exception:
                        pass
                    
                    # Longer waits + WebDriverWait checks to handle lazy loading
                    try:
                        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                        # short scrolls to trigger lazy loads
                        driver.execute_script("window.scrollTo(0, 600);")
                        time.sleep(1)
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        time.sleep(2)
                    except Exception:
                        time.sleep(4)  # fallback slow path

                    edition_html = driver.page_source
                    
                    # Detect "No preview available / can't preview" pages and force the GDrive download flow
                    html_lower = edition_html.lower()
                    if "can't preview file" in html_lower or "no preview available" in html_lower or "google drive - can't open file" in html_lower:
                        # try to extract file id and route to gdrive downloader
                        current_url = driver.current_url
                        fid = self.extract_gdrive_file_id(current_url)
                        if fid:
                            logger.info(f"🔁 Detected Drive preview page without preview; routing to GDrive download for {fid}")
                            return {"source": source_name, "success": True, "file_id": fid, "url": f"https://drive.google.com/uc?id={fid}&export=download", "type": "gdrive", "already_downloaded": False}
                    
                    edition_soup = BeautifulSoup(edition_html, "html.parser")

                    # 1) check iframe/embed/object tags
                    for tag in edition_soup.find_all(['iframe','embed','object']):
                        src = tag.get('src') or tag.get('data-src') or tag.get('data')
                        if src:
                            resolved_src = urljoin(full, src)
                            if resolved_src.lower().endswith('.pdf') or '.pdf' in resolved_src.lower():
                                logger.info(f"🎯 Found PDF in iframe/embed: {resolved_src}")
                                return {"source": source_name, "success": True, "url": resolved_src, "type": "direct_pdf", "already_downloaded": False}

                    # 2) scan anchors for pdfs
                    for ea in edition_soup.find_all("a", href=True):
                        eh = ea['href'].strip()
                        resolved_eh = urljoin(full, eh)
                        if resolved_eh.lower().endswith(".pdf") or ".pdf" in resolved_eh.lower():
                            logger.info(f"🎯 Found PDF anchor: {resolved_eh}")
                            return {"source": source_name, "success": True, "url": resolved_eh, "type": "direct_pdf", "already_downloaded": False}

                    # Aggressive scan of edition_html and scripts for pdf / drive links
                    # 1) direct regex in HTML (already present), but keep and extend:
                    possible = re.findall(r"https?://[^\s'\"<>]+\.pdf", edition_html, flags=re.I)
                    if possible:
                        resolved = urljoin(full, possible[0])
                        logger.info(f"🎯 Found PDF via regex in page source: {resolved}")
                        return {"source": source_name, "success": True, "url": resolved, "type": "direct_pdf", "already_downloaded": False}

                    # 2) find Drive/docs patterns in anchors, data-attrs, and scripts
                    # check anchors first
                    for ea in edition_soup.find_all("a", href=True):
                        candidate = ea['href'].strip()
                        resolved_candidate = self.try_resolve_viewer_link(candidate) or self.try_resolve_viewer_link(urljoin(full, candidate))
                        if resolved_candidate:
                            logger.info(f"🎯 Resolved viewer link from anchor: {resolved_candidate}")
                            return {"source": source_name, "success": True, "url": resolved_candidate, "type": "direct_pdf", "already_downloaded": False}

                    # check data-src / data attributes on tags
                    for tag in edition_soup.find_all(True):
                        for attr in ('data-src','data-href','data','src','data-original'):
                            if tag.has_attr(attr):
                                candidate = tag.get(attr) or ""
                                resolved_candidate = self.try_resolve_viewer_link(candidate) or self.try_resolve_viewer_link(urljoin(full, candidate))
                                if resolved_candidate:
                                    logger.info(f"🎯 Resolved viewer link from tag attr {attr}: {resolved_candidate}")
                                    return {"source": source_name, "success": True, "url": resolved_candidate, "type": "direct_pdf", "already_downloaded": False}

                    # check scripts (inline JS) - sometimes the URL is stored inside JS strings
                    for script in edition_soup.find_all("script"):
                        script_text = script.string or ""
                        resolved_candidate = self.try_resolve_viewer_link(script_text)
                        if resolved_candidate:
                            logger.info(f"🎯 Resolved viewer link from inline script: {resolved_candidate}")
                            return {"source": source_name, "success": True, "url": resolved_candidate, "type": "direct_pdf", "already_downloaded": False}

                    # 4) HEAD check on the edition page itself (some pages redirect to pdf)
                    try:
                        resp = self.session.head(full, allow_redirects=True, timeout=15)
                        ct = resp.headers.get("content-type","").lower()
                        if "pdf" in ct or resp.url.lower().endswith(".pdf"):
                            logger.info(f"🎯 Found PDF via HEAD redirect: {resp.url}")
                            return {"source": source_name, "success": True, "url": resp.url, "type": "direct_pdf", "already_downloaded": False}
                    except Exception as e:
                        logger.debug(f"HEAD check failed for {full}: {e}")

                    # Save edition html for debug if nothing found (helps diagnosis)
                    self.save_debug_page(source_name, edition_html, note="edition_inspect")
                except Exception as e:
                    logger.debug(f"Error following candidate {full}: {e}")

            # if nothing found, dump debug HTML for inspection
            self.save_debug_page(source_name, driver.page_source, note="edition_no_pdf")
            return {"source": source_name, "success": False, "error": "No PDF found in edition pages"}

        except Exception as e:
            logger.error(f"❌ Error preprocessing epaperwave source {source_name}: {e}")
            return {"source": source_name, "success": False, "error": str(e)}
        finally:
            try:
                self.return_driver(driver)
            except Exception:
                pass

    def preprocess_web_source(self, source_name, source_config):
        """Preprocess a web source to get download URL - handles both GDrive and direct PDFs"""
        if source_name in self.already_downloaded:
            return {
                "source": source_name, 
                "success": True, 
                "already_downloaded": True,
                "type": "web"
            }
        
        # Check if this is an epaperwave source
        url = source_config["url"]
        if 'epaperwave.com' in url:
            logger.info(f"🔍 Using specialized epaperwave handler for {source_name}")
            return self.preprocess_epaperwave(source_name, source_config)
        
        driver = self.get_driver(wait_timeout=10)
        if not driver:
            return {"source": source_name, "success": False, "error": "No driver available"}
            
        try:
            date_format = source_config.get("date_format", "%d-%m-%Y")
            
            # Get multiple date variants for better matching
            today_date_variants = self.get_todays_date_formats(date_format, as_list=True)
            
            logger.info(f"🔍 Preprocessing: {source_name} - looking for {today_date_variants[0]} and variants")
            
            # Use try/except for TimeoutException
            try:
                driver.get(url)
                # Log the final URL
                try:
                    cur = driver.current_url
                    logger.info(f"🔗 Selenium ended at URL: {cur}")
                except Exception:
                    pass
                # Wait a bit for Cloudflare / JS checks to complete (configurable)
                logger.debug(f"⏱️ Waiting {self.INITIAL_PAGE_WAIT}s for JS/cloudflare to finish on {url}")
                time.sleep(self.INITIAL_PAGE_WAIT)
                # Then actively wait for cloudflare to finish (up to 30s)
                if not self.wait_for_js_checks(driver, timeout=30):
                    logger.debug("⏱️ Cloudflare/JS check still present after timeout")
            except TimeoutException:
                logger.warning(f"⚠️ driver.get() timed out for {url} — continuing with partial DOM")
            
            # --- PATCH 3: Abort if redirected to Telegram ---
            cur = driver.current_url.lower()
            if "t.me" in cur or "telegram" in cur:
                logger.info(f"❌ Aborted: Page redirected to Telegram → {cur}")
                return {"source": source_name, "success": False, "error": "Redirected to Telegram"}
            
            # quick check: if the driver redirected to a Telegram page, skip this candidate
            try:
                cur = driver.current_url.lower()
                if 't.me' in cur or 'telegram' in cur:
                    logger.info(f"🔕 Page redirected to Telegram ({cur}) — skipping")
                    return {"source": source_name, "success": False, "error": "Page redirected to Telegram"}
            except Exception:
                pass
            
            # Dismiss overlays before processing
            self.dismiss_overlays(driver)
            
            time.sleep(3)
            
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Broader regex-based date search
            # Broader date matching: numeric day, month names, and short forms (case-insensitive)
            today = datetime.now()
            day = today.day
            year = today.year
            month_long = today.strftime("%B")
            month_short = today.strftime("%b")

            # build several regex candidates (yearless included)
            date_patterns = [
                rf"\b{day}\b",                               # day number anywhere
                rf"\b{day}\s*[-/]\s*{today.strftime('%m')}\b",
                rf"\b{day}\s*{re.escape(month_long)}\b",
                rf"\b{day}\s*{re.escape(month_short)}\b",
                rf"{re.escape(month_long)}\s*{day}",
                rf"{re.escape(month_short)}\s*{day}",
                rf"{re.escape(month_long)}\b",               # month header fallback
                rf"{re.escape(month_short)}\b",
            ]

            date_regex = re.compile("|".join(date_patterns), flags=re.I)
            date_elements = soup.find_all(string=lambda text: bool(text and date_regex.search(str(text))))
            
            if not date_elements:
                # More aggressive site-scan fallback
                # aggressive site-scan: look for any link that contains 'epaper' or '/paper/' and try it
                found_epaper_link = None
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    if 'epaper' in href.lower() or '/paper/' in href.lower() or '/edition/' in href.lower():
                        resolved = urljoin(url, href)
                        logger.info(f"🔎 Aggressive fallback trying {resolved}")
                        # attempt a quick HEAD or short GET to see if it's a PDF or redirect
                        try:
                            r = self.session.head(resolved, allow_redirects=True, timeout=8)
                            # Handle 403 with GET retry
                            if r.status_code == 403:
                                # try with a browser-like GET + referer & cookies (some hosts block HEAD)
                                logger.info("🔁 HEAD 403; retrying GET with browser headers")
                                r = self.session.get(resolved, timeout=12, allow_redirects=True, headers={"Referer": url, "User-Agent": self.session.headers.get("User-Agent")})
                            if (r.headers.get('content-type','') and 'pdf' in r.headers.get('content-type','').lower()) or r.url.lower().endswith('.pdf'):
                                found_epaper_link = r.url
                                logger.info(f"✅ Found PDF via aggressive scan: {found_epaper_link}")
                                return {"source": source_name, "success": True, "url": found_epaper_link, "type": "direct_pdf", "already_downloaded": False}
                        except Exception:
                            pass
                
                # Aggressive attribute + JS-string PDF detection
                # collect extra candidates from data-* attrs, onclicks, and scripts
                extra_candidates = set()
                for tag in soup.find_all(True):
                    for attr in ('data-href','data-src','data','data-original','data-url','onclick','href','src'):
                        if tag.has_attr(attr):
                            val = (tag.get(attr) or "").strip()
                            if val:
                                # possibly contains encoded or direct pdf/drive url
                                if '.pdf' in val.lower() or 'drive.google.com' in val.lower() or 'uc?export=download' in val.lower():
                                    extra_candidates.add(urljoin(url, val))
                # Check inline scripts
                for script in soup.find_all("script"):
                    st = script.string or ""
                    if st and ('.pdf' in st.lower() or 'drive.google.com' in st.lower()):
                        m = re.search(r"https?://[^\s'\"<>]+\.pdf", st, flags=re.I)
                        if m:
                            extra_candidates.add(m.group(0))
                        # also look for drive patterns
                        m2 = re.search(r"/uc\?id=[A-Za-z0-9_\-]+", st)
                        if m2:
                            extra_candidates.add(urljoin("https://drive.google.com", m2.group(0)))
                # Try extra candidates first
                for cand in extra_candidates:
                    logger.info(f"🔎 Extra candidate from attr/JS: {cand}")
                    # Skip Telegram links
                    if 't.me' in cand or 'telegram' in cand:
                        logger.info(f"🔕 Skipping telegram link from attr/JS: {cand}")
                        continue
                    link_type = self.detect_link_type(cand)
                    if link_type == "gdrive":
                        file_id = self.extract_gdrive_file_id(cand)
                        if file_id:
                            return {"source": source_name, "success": True, "file_id": file_id, "url": cand, "type": "gdrive", "already_downloaded": False}
                        else:
                            return {"source": source_name, "success": True, "url": cand, "type": "direct_pdf", "already_downloaded": False}
                    elif link_type in ["direct_pdf", "likely_pdf"]:
                        return {"source": source_name, "success": True, "url": cand, "type": "direct_pdf", "already_downloaded": False}
                
                # Original fallback scanning
                anchors = soup.find_all('a', href=True)
                for a in anchors:
                    txt = (a.get_text() or "").lower()
                    href = a['href']
                    if any(k in txt for k in ('edition','editions','download','epaper','view','pdf')) or '.pdf' in href.lower():
                        resolved = urljoin(url, href)
                        logger.info(f"🔎 Fallback pick -> {resolved} (anchor text: {txt[:80]})")
                        link_type = self.detect_link_type(resolved)
                        if link_type == "gdrive":
                            file_id = self.extract_gdrive_file_id(resolved)
                            if file_id:
                                return {"source": source_name, "success": True, "file_id": file_id, "url": resolved, "type": "gdrive", "already_downloaded": False}
                            else:
                                return {"source": source_name, "success": True, "url": resolved, "type": "direct_pdf", "already_downloaded": False}
                        else:
                            return {"source": source_name, "success": True, "url": resolved, "type": "direct_pdf", "already_downloaded": False}
                # still not found -> save debug and return
                self.save_debug_page(source_name, driver.page_source, note="no_date_no_fallback")
                return {"source": source_name, "success": False, "error": "Date not found and no fallback link"}
            
            # Find download link near the date
            download_link = None
            for date_element in date_elements:
                parent = date_element.parent
                max_lookahead = 5
                
                current = parent
                for i in range(max_lookahead):
                    if current:
                        links = current.find_all('a', href=True)
                        for link in links:
                            link_text_actual = link.get_text().strip().lower()
                            href = link['href']
                            
                            # Skip Telegram links
                            if 't.me' in href or 'telegram' in href:
                                logger.info(f"🔕 Skipping telegram link while preprocessing: {href}")
                                continue
                            
                            # More flexible matching for link text
                            if (any(keyword in link_text_actual for keyword in ['click here', 'download', 'pdf']) and 
                                href != '#' and 
                                'http' in href):
                                download_link = href
                                logger.info(f"🎯 Found download link: {link_text_actual} -> {href}")
                                break
                        
                        if download_link:
                            break
                    
                    if hasattr(current, 'next_sibling'):
                        current = current.next_sibling
                    else:
                        break
                
                if download_link:
                    break

            if not download_link:
                return {"source": source_name, "success": False, "error": "Download link not found"}
            
            # Skip Telegram links
            if 't.me' in download_link or 'telegram' in download_link:
                logger.info(f"🔕 Skipping telegram link: {download_link}")
                return {"source": source_name, "success": False, "error": "Telegram link, skipping"}
            
            # Auto-detect link type and process accordingly
            link_type = self.detect_link_type(download_link)
            logger.info(f"🔗 Link type detected: {link_type} for {source_name}")
            
            if link_type == "gdrive":
                file_id = self.extract_gdrive_file_id(download_link)
                if file_id:
                    return {
                        "source": source_name,
                        "success": True,
                        "file_id": file_id,
                        "download_url": download_link,
                        "type": "gdrive",
                        "already_downloaded": False
                    }
                else:
                    return {"source": source_name, "success": False, "error": "Could not extract Google Drive ID"}
            
            elif link_type in ["direct_pdf", "likely_pdf"]:
                return {
                    "source": source_name,
                    "success": True,
                    "url": download_link,
                    "type": "direct_pdf",
                    "already_downloaded": False
                }
            
            else:
                # For unknown links, try to follow them to see if they lead to PDFs
                logger.info(f"🔄 Following unknown link type for {source_name}")
                try:
                    # Use try/except for TimeoutException
                    try:
                        driver.get(download_link)
                        # Log the final URL
                        try:
                            cur = driver.current_url
                            logger.info(f"🔗 Selenium ended at URL: {cur}")
                        except Exception:
                            pass
                        # WAIT on first navigation to allow checks to finish
                        logger.debug(f"⏱️ Waiting {self.INITIAL_PAGE_WAIT}s after opening candidate {download_link}")
                        time.sleep(self.INITIAL_PAGE_WAIT)
                        # Then actively wait for cloudflare to finish
                        if not self.wait_for_js_checks(driver, timeout=30):
                            logger.debug("⏱️ Cloudflare/JS check still present after timeout")
                    except TimeoutException:
                        logger.warning(f"⚠️ driver.get() timed out for {download_link} — continuing with partial DOM")
                    
                    # quick check: if the driver redirected to a Telegram page, skip this candidate
                    try:
                        cur = driver.current_url.lower()
                        if 't.me' in cur or 'telegram' in cur:
                            logger.info(f"🔕 candidate redirected to Telegram ({cur}) — skipping")
                            return {"source": source_name, "success": False, "error": "Redirected to Telegram"}
                    except Exception:
                        pass
                    
                    # Dismiss overlays before processing
                    self.dismiss_overlays(driver)
                    
                    # --- Check for iframes that point to Google Drive viewers and use that src directly ---
                    try:
                        frames = driver.find_elements(By.TAG_NAME, "iframe")
                        for fr in frames:
                            try:
                                src = (fr.get_attribute("src") or fr.get_attribute("data-src") or "").strip()
                                if src:
                                    resolved_src = urljoin(download_link, src)
                                    fid = self.extract_gdrive_file_id(resolved_src)
                                    if fid:
                                        logger.info(f"🔎 Found drive iframe src; routing to drive handler: {resolved_src}")
                                        return {"source": source_name, "success": True, "file_id": fid, "url": resolved_src, "type": "gdrive", "already_downloaded": False}
                                    elif resolved_src.lower().endswith('.pdf') or '.pdf' in resolved_src.lower():
                                        return {"source": source_name, "success": True, "url": resolved_src, "type": "direct_pdf", "already_downloaded": False}
                            except Exception:
                                continue
                    except Exception:
                        pass
                    
                    time.sleep(2)
                    
                    current_url = driver.current_url
                    
                    # Detect "No preview available / can't preview" pages and force the GDrive download flow
                    body = driver.page_source
                    html_lower = body.lower()
                    if "can't preview file" in html_lower or "no preview available" in html_lower or "google drive - can't open file" in html_lower:
                        # try to extract file id and route to gdrive downloader
                        fid = self.extract_gdrive_file_id(current_url)
                        if fid:
                            logger.info(f"🔁 Detected Drive preview page without preview; routing to GDrive download for {fid}")
                            return {"source": source_name, "success": True, "file_id": fid, "url": f"https://drive.google.com/uc?id={fid}&export=download", "type": "gdrive", "already_downloaded": False}
                    
                    final_link_type = self.detect_link_type(current_url)
                    
                    if final_link_type in ["direct_pdf", "likely_pdf"]:
                        return {
                            "source": source_name,
                            "success": True,
                            "url": current_url,
                            "type": "direct_pdf",
                            "already_downloaded": False
                        }
                    elif final_link_type == "gdrive":
                        file_id = self.extract_gdrive_file_id(current_url)
                        if file_id:
                            return {
                                "source": source_name,
                                "success": True,
                                "file_id": file_id,
                                "download_url": current_url,
                                "type": "gdrive",
                                "already_downloaded": False
                            }
                    
                    return {"source": source_name, "success": False, "error": f"Unknown link type: {final_link_type}"}
                    
                except Exception as e:
                    return {"source": source_name, "success": False, "error": f"Error following link: {str(e)}"}
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing {source_name}: {e}")
            return {"source": source_name, "success": False, "error": str(e)}
        finally:
            if driver:
                self.return_driver(driver)

    def preprocess_direct_pdf_source(self, source_name, source_config):
        """Preprocess direct PDF source with predictable URL patterns"""
        if source_name in self.already_downloaded:
            return {
                "source": source_name, 
                "success": True, 
                "already_downloaded": True,
                "type": "direct_pdf"
            }
            
        try:
            url_pattern = source_config["url_pattern"]
            date_format = source_config.get("date_format", "%d%m%Y")
            
            today_date = self.get_todays_date_formats(date_format, as_list=False)
            url = url_pattern.format(date=today_date)
            
            return {
                "source": source_name,
                "success": True,
                "url": url,
                "type": "direct_pdf",
                "already_downloaded": False
            }
                
        except Exception as e:
            logger.error(f"❌ Error preprocessing direct PDF source {source_name}: {e}")
            return {"source": source_name, "success": False, "error": str(e)}

    def preprocess_all_sources_parallel(self):
        """Preprocess all sources in parallel"""
        preprocessing_tasks = []
        results = []
        
        # Add web sources (both GDrive and direct PDFs via websites)
        for source_name, source_config in self.web_sources.items():
            if source_name not in self.already_downloaded:
                preprocessing_tasks.append((source_name, source_config, "web"))
            else:
                logger.info(f"✅ Skipping {source_name} - already downloaded")
        
        # Add direct PDF sources with predictable URLs
        direct_pdf_sources = self.config.get("direct_pdf_sources", {})
        for source_name, source_config in direct_pdf_sources.items():
            if source_name not in self.already_downloaded:
                preprocessing_tasks.append((source_name, source_config, "direct_pdf"))
            else:
                logger.info(f"✅ Skipping {source_name} - already downloaded")
        
        if not preprocessing_tasks:
            logger.info("ℹ️ No sources need preprocessing - all already downloaded")
            return []
        
        logger.info(f"🚀 Starting parallel preprocessing of {len(preprocessing_tasks)} sources")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_preprocess_workers) as executor:
            future_to_task = {}
            
            for source_name, source_config, source_type in preprocessing_tasks:
                if source_type == "web":
                    future = executor.submit(self.preprocess_web_source, source_name, source_config)
                else:
                    future = executor.submit(self.preprocess_direct_pdf_source, source_name, source_config)
                
                future_to_task[future] = (source_name, source_type)
            
            for future in concurrent.futures.as_completed(future_to_task):
                source_name, source_type = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result["success"]:
                        if result.get("already_downloaded", False):
                            logger.info(f"✅ Already downloaded: {source_name}")
                        else:
                            logger.info(f"✅ Preprocessing successful: {source_name}")
                    else:
                        logger.warning(f"⚠️ Preprocessing failed: {source_name} - {result.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"❌ Preprocessing error for {source_name}: {e}")
                    results.append({"source": source_name, "success": False, "error": str(e)})
        
        return results

    def download_batch_serial(self, batch):
        """Download a batch of preprocessed sources serially"""
        successful_downloads = []
        
        for item in batch:
            if not item["success"] or item.get("already_downloaded", False):
                continue
                
            source_name = item["source"]
            
            try:
                if item["type"] == "gdrive":
                    # Extract resourcekey from the discovered URL (if any) so we can pass it through.
                    resource_key = self.extract_gdrive_resource_key(item.get("url", ""))
                    success = self.download_gdrive_file(item["file_id"], source_name, resource_key=resource_key)
                else:  # direct_pdf
                    success = self.download_direct_file(item["url"], source_name)
                
                if success:
                    successful_downloads.append(source_name)
                    status(f"✅ Downloaded {source_name}")
                    logger.info(f"✅ Download completed: {source_name}")
                else:
                    # try to report any nearby debug info: files in debug dir, or the http status if available
                    logger.error(f"❌ Download failed: {source_name} — check newspaper_downloader.log and debug/ directory")
                    
            except Exception as e:
                logger.error(f"❌ Error downloading {source_name}: {e}")
        
        return successful_downloads

    def download_gdrive_file_via_selenium(self, file_id, source_name, resource_key=None, wait_for=8):
        """Selenium-assisted GDrive download: click download button, copy cookies, then use requests."""
        driver = self.get_driver(wait_timeout=10)
        if not driver:
            logger.debug("No selenium driver available for GDrive selenium fallback")
            return False

        success = False
        try:
            base_view = f"https://drive.google.com/uc?id={file_id}&export=download"
            if resource_key:
                base_view += f"&resourcekey={urllib.parse.quote_plus(resource_key)}"
            logger.info(f"🧪 Selenium fallback: opening {base_view}")
            try:
                driver.get(base_view)
                # Wait for Cloudflare/JS checks
                time.sleep(1)
                if not self.wait_for_js_checks(driver, timeout=30):
                    logger.debug("⏱️ Cloudflare/JS check still present after timeout")
            except TimeoutException:
                logger.warning("⚠️ Selenium get() timed out for gdrive fallback; continuing")

            time.sleep(1)
            self.dismiss_overlays(driver)
            
            # HARD REMOVE Google Drive blocking overlays
            try:
                driver.execute_script("""
                    document.querySelectorAll('div[class*="ndfHFb"]').forEach(e => e.remove());
                    document.querySelectorAll('div[role="dialog"]').forEach(e => e.remove());
                    document.querySelectorAll('div[jscontroller]').forEach(e => {
                        if (e.style && e.style.zIndex) e.remove();
                    });
                """)
            except:
                pass
            
            # ------------------------------------------------------------
            # NEW: Google Drive "download-form" detection and submission
            # ------------------------------------------------------------
            try:
                # 1) quick: detect and submit the Drive "download-form" directly (most robust)
                form = None
                try:
                    form = driver.find_element(By.ID, "download-form")
                except Exception:
                    # maybe different id; try generic search for uc-main form
                    try:
                        form = driver.find_element(By.XPATH, "//form[contains(@id,'download') or contains(@class,'uc-download') or contains(., 'uc-download-link')]")
                    except Exception:
                        form = None

                if form:
                    logger.info("🔎 Found download form -> extracting action + hidden inputs")
                    action = form.get_attribute("action") or driver.current_url
                    method = (form.get_attribute("method") or "get").lower()

                    # collect inputs (hidden / text / values)
                    params = {}
                    try:
                        inputs = form.find_elements(By.XPATH, ".//input")
                        for inp in inputs:
                            name = inp.get_attribute("name")
                            if not name:
                                continue
                            val = inp.get_attribute("value") or ""
                            params[name] = val
                    except Exception:
                        pass

                    # build final URL for GET submission (if method is GET)
                    if method == "get":
                        final_url = action
                        if params:
                            final_url = final_url + ("&" if "?" in final_url else "?") + urllib.parse.urlencode(params)
                        logger.info(f"🔎 Form GET -> {final_url}")

                        # copy selenium cookies into requests session and try GET
                        req_sess = requests.Session()
                        req_sess.headers.update(self.session.headers)
                        self.copy_selenium_cookies_to_session(driver, req_sess)

                        try:
                            r = req_sess.get(final_url, timeout=120, stream=True, headers={"Referer": driver.current_url})
                            if r.status_code == 200 and ('pdf' in (r.headers.get('content-type') or '').lower() or r.url.lower().endswith('.pdf')):
                                download_dir = Path(self.config["settings"]["download_dir"])
                                today_str = datetime.now().strftime("%Y%m%d")
                                safe_name = sanitize_filename(f"{source_name}_{today_str}.pdf")
                                out_file = download_dir / safe_name
                                
                                # Use atomic write
                                try:
                                    atomic_write_stream(r, out_file, chunk_size=8192)
                                except Exception as e:
                                    logger.debug(f"Atomic write failed: {e}")
                                    return False
                                
                                # Use finalize_download for validation
                                if self.finalize_download(out_file, source_name):
                                    logger.info(f"✅ GDrive form GET download successful: {safe_name}")
                                    success = True
                                    return True
                                else:
                                    logger.debug("⚠️ Form GET returned non-PDF or invalid file; falling back")
                        except Exception as e:
                            logger.debug(f"Form GET attempt failed: {e}")

                    # If GET path didn't succeed, submit the form via JS (this triggers the 'download anyway' action)
                    try:
                        logger.info("🔁 Submitting download form via JS (driver.execute_script)")
                        driver.execute_script("document.getElementById('download-form') && document.getElementById('download-form').submit();")
                        # give it a moment to redirect to the real download / confirm page
                        time.sleep(3)
                        # if it redirected to a .pdf or to an URL with confirm token, let main logic catch it afterwards
                    except Exception as e:
                        logger.debug(f"Form submit via JS failed: {e}")
            except Exception as e:
                logger.debug(f"Form-detection helper failed silently: {e}")

            # ------------------------------------------------------------
            # IMPROVED UNIVERSAL GOOGLE DRIVE "DOWNLOAD ANYWAY" CLICKER
            # ------------------------------------------------------------
            
            download_clicked = False
            wait = WebDriverWait(driver, 20)  # Increased wait time

            # Updated candidates_xpaths with additional input element selectors
            candidates_xpaths = [
                "//a[@id='uc-download-link']",
                "//a[contains(@href,'confirm=')]",
                "//input[@id='uc-download-link']",
                "//input[@type='submit' and (contains(translate(@value,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'download') or contains(translate(@value,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'download anyway'))]",
                "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'download anyway')]",
                "//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'download anyway')]",
                "//*[@role='button' and contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), 'download')]",
                # ADDED: Extra TOI-specific selectors
                "//a[contains(@href,'confirm=') and contains(@href,'uc?export=download')]",
                "//a[contains(@class,'download') and contains(@href,'uc?export=download')]",
                "//button[contains(@class,'download') or contains(@class,'uc-download')]",
                "//div[contains(@id,'download') or contains(@class,'download')]/a"
            ]

            for xp in candidates_xpaths:
                try:
                    elem = wait.until(EC.element_to_be_clickable((By.XPATH, xp)))
                    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
                    time.sleep(0.3)
                    try:
                        elem.click()
                    except Exception:
                        driver.execute_script("arguments[0].click();", elem)
                    download_clicked = True
                    time.sleep(2)  # Wait after clicking
                    
                    # Try to detect browser download after clicking
                    browser_saved_file = self._move_latest_browser_download(source_name, wait_seconds=18)
                    if browser_saved_file:
                        # We moved it to the canonical filename; validate + finalize like other flows
                        if self.finalize_download(browser_saved_file, source_name):
                            logger.info(f"✅ Selenium-assisted browser download successful (moved): {browser_saved_file.name}")
                            success = True
                            return True
                        else:
                            logger.warning("⚠️ Browser download saved but failed PDF validation")
                            # optionally mark skip or continue with other fallbacks
                    
                    break
                except TimeoutException:
                    continue

            if not download_clicked:
                # fallback: look for any link that contains 'confirm=' in page source and open it
                page = driver.page_source
                m = re.search(r'(/uc\?export=download[^"\']+confirm[^"\']+)', page)
                if m:
                    confirm_url = urllib.parse.unquote(m.group(1))
                    confirm_url = urljoin("https://drive.google.com", confirm_url)
                    # try requests GET with cookies copied from driver
                    req_sess = requests.Session()
                    req_sess.headers.update(self.session.headers)
                    self.copy_selenium_cookies_to_session(driver, req_sess)
                    headers = {"Referer": "https://drive.google.com", "User-Agent": self.session.headers.get("User-Agent")}
                    resp = req_sess.get(confirm_url, timeout=120, stream=True, headers=headers)
                    if resp.status_code == 200:
                        # save the file
                        download_dir = Path(self.config["settings"]["download_dir"])
                        today_str = datetime.now().strftime("%Y%m%d")
                        safe_name = sanitize_filename(f"{source_name}_{today_str}.pdf")
                        out_file = download_dir / safe_name
                        
                        # Use atomic write
                        try:
                            atomic_write_stream(resp, out_file, chunk_size=8192)
                        except Exception as e:
                            logger.debug(f"Atomic write failed: {e}")
                            return False
                        
                        if self.finalize_download(out_file, source_name):
                            logger.info(f"✅ GDrive download (selenium-assisted) successful: {safe_name}")
                            success = True
                            return True
                        else:
                            self.mark_skip_gdrive(file_id, reason="invalid_post_download", hours=24)
                            return False

            # SECOND-STAGE VIRUS-SCAN WARNING PAGE
            try:
                btn2 = driver.find_element(By.XPATH, "//button[contains(., 'Download anyway')]")
                driver.execute_script("arguments[0].scrollIntoView(true);", btn2)
                time.sleep(0.3)
                btn2.click()
                logger.info("🔎 Clicked second-stage 'Download anyway'")
                
                # Also try to detect browser download after clicking second-stage button
                browser_saved_file = self._move_latest_browser_download(source_name, wait_seconds=18)
                if browser_saved_file:
                    if self.finalize_download(browser_saved_file, source_name):
                        logger.info(f"✅ Selenium-assisted browser download successful (second stage): {browser_saved_file.name}")
                        success = True
                        return True
                
                time.sleep(2)
            except:
                pass

            # Wait for the confirm/download element after clicking
            try:
                WebDriverWait(driver, 10).until(lambda d: (
                    'uc?export=download' in (d.page_source or "").lower()
                    or 'confirm=' in (d.page_source or "").lower()
                    or d.current_url.lower().endswith('.pdf')
                ))
            except Exception:
                # proceed — we already have fallback paths
                logger.debug("⏱️ wait for confirm/download element timed out; continuing with heuristics")

            # Wait a moment for redirect or for the confirm URL
            time.sleep(2)
            
            # ------------------------------------------------------------
            # TOI-ROBUST CONFIRM TOKEN EXTRACTION
            # ------------------------------------------------------------
            page = driver.page_source
            confirm_url = None

            # Pattern 1 — normal Drive confirm URL
            m = re.search(r'href="(/uc\?export=download[^"]+confirm[^"]+)"', page, flags=re.I)
            if m:
                confirm_path = urllib.parse.unquote(m.group(1))
                confirm_url = urljoin("https://drive.google.com", confirm_path)
                logger.info(f"🔎 Confirm URL (href) → {confirm_url}")

            # Pattern 2 — JS-based confirm URL
            if not confirm_url:
                m2 = re.search(r'"(\/uc\?export=download[^"]+confirm[^"]+)"', page, flags=re.I)
                if m2:
                    confirm_path = urllib.parse.unquote(m2.group(1))
                    confirm_url = urljoin("https://drive.google.com", confirm_path)
                    logger.info(f"🔎 Confirm URL (JS) → {confirm_url}")

            # Pattern 3 — extremely hidden token, TOI-specific
            if not confirm_url:
                m3 = re.search(r'confirm=([0-9A-Za-z_\-]+)', page)
                if m3:
                    token = m3.group(1)
                    confirm_url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm={token}"
                    if resource_key:
                        confirm_url += f"&resourcekey={urllib.parse.quote_plus(resource_key)}"
                    logger.info(f"🔎 Confirm token only → {confirm_url}")

            # Pattern 4 — check current_url params for token
            if not confirm_url:
                try:
                    parsed = urllib.parse.urlparse(driver.current_url)
                    params = urllib.parse.parse_qs(parsed.query)
                    if 'confirm' in params:
                        token = params['confirm'][0]
                        confirm_url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm={token}"
                        if resource_key:
                            confirm_url += f"&resourcekey={urllib.parse.quote_plus(resource_key)}"
                        logger.info(f"🔎 Found confirm token in current_url -> {confirm_url}")
                except Exception:
                    pass

            if confirm_url:
                # copy cookies from selenium to requests with robust domain handling
                req_sess = requests.Session()
                req_sess.headers.update(self.session.headers)
                self.copy_selenium_cookies_to_session(driver, req_sess)
                try:
                    headers = {"Referer": "https://drive.google.com", "User-Agent": self.session.headers.get("User-Agent")}
                    resp = req_sess.get(confirm_url, timeout=120, stream=True, headers=headers)
                    if resp.status_code == 200:
                        # write to file
                        download_dir = Path(self.config["settings"]["download_dir"])
                        today_str = datetime.now().strftime("%Y%m%d")
                        safe_name = sanitize_filename(f"{source_name}_{today_str}.pdf")
                        out_file = download_dir / safe_name
                        
                        # Use atomic write
                        try:
                            atomic_write_stream(resp, out_file, chunk_size=8192)
                        except Exception as e:
                            logger.debug(f"Atomic write failed: {e}")
                            return False
                        
                        # Validate the downloaded file
                        if self.finalize_download(out_file, source_name):
                            logger.info(f"✅ GDrive download (selenium-assisted) successful: {safe_name}")
                            success = True
                            return True
                        else:
                            self.mark_skip_gdrive(file_id, reason="invalid_post_download", hours=24)
                            return False
                except Exception as e:
                    logger.debug(f"Selenium-assisted confirm GET failed: {e}")

            # If still not found, try extracting cookies and calling uc?id=...&export=download directly
            req_sess = requests.Session()
            req_sess.headers.update(self.session.headers)
            self.copy_selenium_cookies_to_session(driver, req_sess)
            final_url = f"https://drive.google.com/uc?id={file_id}&export=download"
            if resource_key:
                final_url += f"&resourcekey={urllib.parse.quote_plus(resource_key)}"
            try:
                headers = {"Referer": "https://drive.google.com", "User-Agent": self.session.headers.get("User-Agent")}
                resp = req_sess.get(final_url, timeout=120, stream=True, allow_redirects=True, headers=headers)
                if resp.status_code == 200 and ('pdf' in (resp.headers.get('content-type') or '').lower() or resp.url.lower().endswith('.pdf') or (resp.headers.get('content-length') and int(resp.headers.get('content-length'))>self.MIN_ACCEPTED_FILESIZE)):
                    download_dir = Path(self.config["settings"]["download_dir"])
                    today_str = datetime.now().strftime("%Y%m%d")
                    safe_name = sanitize_filename(f"{source_name}_{today_str}.pdf")
                    out_file = download_dir / safe_name
                    
                    # Use atomic write
                    try:
                        atomic_write_stream(resp, out_file, chunk_size=8192)
                    except Exception as e:
                        logger.debug(f"Atomic write failed: {e}")
                        return False
                    
                    # Validate the downloaded file
                    if self.finalize_download(out_file, source_name):
                        logger.info(f"✅ GDrive download (selenium cookie) successful: {safe_name}")
                        success = True
                        return True
                    else:
                        self.mark_skip_gdrive(file_id, reason="invalid_post_download", hours=24)
                        return False
            except Exception as e:
                logger.debug(f"Selenium cookie GET failed: {e}")

            logger.debug("Selenium-assisted GDrive fallback did not find a downloadable PDF")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error in selenium fallback for {source_name}: {e}")
            return False
            
        finally:
            if not success and driver is not None:
                # Save debug info for failure analysis
                try:
                    debug_dir = Path(self.config["settings"]["download_dir"]) / "debug"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    driver.save_screenshot(str(debug_dir / sanitize_filename(f"{source_name}_gdrive_failed_{timestamp}.png")))
                    (debug_dir / sanitize_filename(f"{source_name}_gdrive_failed_{timestamp}.html")).write_text(driver.page_source, encoding="utf-8")
                    logger.info(f"📸 Saved failure debug for {source_name} to {debug_dir}")
                except Exception as e:
                    logger.debug(f"Could not save debug screenshot/html: {e}")
            
            # always return driver
            try:
                self.return_driver(driver)
            except Exception:
                pass

    def download_gdrive_file(self, file_id, source_name, resource_key=None):
        """Download a file from Google Drive with confirm token handling."""
        with self.download_lock:
            # don't attempt if recently marked skip
            if self.should_skip_gdrive(file_id):
                logger.info(f"⏸️ Skipping GDrive {file_id} because it was marked to retry later")
                return False
                
            download_dir = Path(self.config["settings"]["download_dir"])
            today_str = datetime.now().strftime("%Y%m%d")
            safe_name = sanitize_filename(f"{source_name}_{today_str}.pdf")
            out_file = download_dir / safe_name

            try:
                base = "https://drive.google.com/uc"
                params = {"id": file_id, "export": "download"}
                initial_url = f"{base}?id={file_id}&export=download"
                if resource_key:
                    initial_url += f"&resourcekey={urllib.parse.quote_plus(resource_key)}"

                logger.info(f"⬇️ Starting GDrive download: {safe_name} ({initial_url})")
                resp = self.session.get(initial_url, timeout=60, stream=True, allow_redirects=True)
                
                # Extra: log/inspect the HTML body length and first 400 chars when GDrive returns 200 but no PDF
                logger.debug(f"GDrive initial GET: status={resp.status_code} content-type={resp.headers.get('content-type')} url={resp.url} body_len={len(resp.text or '')}")
                logger.debug((resp.text or '')[:400])  # first 400 chars for quick inspection

                # If response looks like a PDF, save it immediately
                ct = resp.headers.get("content-type", "").lower()
                clen = resp.headers.get("content-length")
                if 'pdf' in ct or (clen and int(clen) > self.MIN_ACCEPTED_FILESIZE) or resp.url.lower().endswith('.pdf'):
                    # Use atomic write
                    try:
                        atomic_write_stream(resp, out_file, chunk_size=8192)
                    except Exception as e:
                        logger.debug(f"Atomic write failed: {e}")
                        return False
                    
                    # Use finalize_download for validation
                    if self.finalize_download(out_file, source_name):
                        logger.info(f"✅ GDrive direct download successful: {safe_name} ({out_file.stat().st_size} bytes)")
                        return True
                    else:
                        self.mark_skip_gdrive(file_id, reason="invalid_after_direct", hours=24)
                        return False

                # Otherwise we probably got the 'can't preview' page. Try to find confirm token.
                body = resp.text or ""
                body_lower = body.lower()

                # --- PATCH 1: Handle Google Drive 'No preview / Download anyway' reliably ---
                if ("download anyway" in body_lower 
                    or "can't preview" in body_lower 
                    or "can't scan" in body_lower 
                    or "virus" in body_lower):
                    
                    # Try to find confirm=TOKEN inside JS
                    m_confirm = re.search(r'"(\/uc\?export=download[^"]+confirm[^"]+)"', body)
                    if m_confirm:
                        confirm_url = urljoin("https://drive.google.com", urllib.parse.unquote(m_confirm.group(1)))
                        logger.info(f"🔎 Found Drive confirm download-anyway URL → {confirm_url}")
                        headers = {"Referer": "https://drive.google.com", "User-Agent": self.session.headers.get("User-Agent")}
                        r2 = self.session.get(confirm_url, timeout=60, stream=True, headers=headers)
                        if r2.status_code == 200:
                            # Use atomic write
                            try:
                                atomic_write_stream(r2, out_file, chunk_size=8192)
                            except Exception as e:
                                logger.debug(f"Atomic write failed: {e}")
                                return False
                            
                            if self.finalize_download(out_file, source_name):
                                logger.info("✅ GDrive 'download anyway' PDF validated successfully")
                                return True
                            else:
                                self.mark_skip_gdrive(file_id, "invalid_after_download_anyway", hours=24)
                                return False

                    logger.info("🔁 Falling back to Selenium-assisted download (download_gdrive_file_via_selenium)")
                    return self.download_gdrive_file_via_selenium(file_id, source_name, resource_key=resource_key)

                # === Enhanced confirm-token / download-anyway detection ===
                # quick heuristics: Drive 'can't scan' message or 'download anyway'
                if ("can't scan" in body_lower or "can't preview file" in body_lower or "download anyway" in body_lower or "virus" in body_lower):
                    logger.info("⚠️ Drive preview page indicates 'can't scan' / 'download anyway' flow — trying to extract confirm token or fallback to selenium")

                # try patterns commonly used by Drive embedded JS or HTML
                # 1) /uc?export=download... in hrefs (existing)
                m = re.search(r'href="(/uc\?export=download[^"]+)"', body)
                if m:
                    confirm_path = urllib.parse.unquote(m.group(1))
                    confirm_url = urljoin("https://drive.google.com", confirm_path)
                    logger.info(f"🔎 Found confirm download link in HTML -> {confirm_url}")
                    headers = {"Referer": "https://drive.google.com", "User-Agent": self.session.headers.get("User-Agent")}
                    confirmed = self.session.get(confirm_url, timeout=60, stream=True, headers=headers)
                    if confirmed.status_code == 200:
                        # Use atomic write
                        try:
                            atomic_write_stream(confirmed, out_file, chunk_size=8192)
                        except Exception as e:
                            logger.debug(f"Atomic write failed: {e}")
                            return False
                        
                        if self.finalize_download(out_file, source_name):
                            logger.info(f"✅ GDrive download successful after confirm: {safe_name} ({out_file.stat().st_size} bytes)")
                            return True
                        else:
                            self.mark_skip_gdrive(file_id, reason="invalid_after_confirm", hours=24)
                            return False

                # 2) tokens like confirm=XYZ in JS (existing)
                m2 = re.search(r'confirm=([0-9A-Za-z_\-]+)&id=', body)
                if m2:
                    token = m2.group(1)
                    final = f"{base}?id={file_id}&export=download&confirm={token}"
                    if resource_key:
                        final += f"&resourcekey={urllib.parse.quote_plus(resource_key)}"
                    logger.info(f"🔎 Found confirm token -> {final}")
                    headers = {"Referer": "https://drive.google.com", "User-Agent": self.session.headers.get("User-Agent")}
                    final_resp = self.session.get(final, timeout=60, stream=True, headers=headers)
                    if final_resp.status_code == 200:
                        # Use atomic write
                        try:
                            atomic_write_stream(final_resp, out_file, chunk_size=8192)
                        except Exception as e:
                            logger.debug(f"Atomic write failed: {e}")
                            return False
                        
                        if self.finalize_download(out_file, source_name):
                            logger.info(f"✅ GDrive download successful after token: {safe_name} ({out_file.stat().st_size} bytes)")
                            return True
                        else:
                            self.mark_skip_gdrive(file_id, reason="invalid_after_token", hours=24)
                            return False

                # Additional regex patterns for confirm token extraction
                # EXTRA: more patterns for confirm token (handles 'confirm': 'TOKEN', confirm = "TOKEN", confirmTOKEN variants)
                confirm_url = None
                m_extra = re.search(r"""(?:
                    confirm\s*[:=]\s*['"]?([0-9A-Za-z_\-]+)['"]?|        # confirm: 'TOKEN' or confirm = "TOKEN"
                    ['"]confirm['"]\s*[:=]\s*['"]?([0-9A-Za-z_\-]+)['"]?   # 'confirm': 'TOKEN' style
                )""", body, flags=re.I | re.X)
                if m_extra:
                    token = next((g for g in m_extra.groups() if g), None)
                    if token:
                        confirm_url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm={token}"
                        if resource_key:
                            confirm_url += f"&resourcekey={urllib.parse.quote_plus(resource_key)}"
                        logger.info(f"🔎 Extra confirm token found → {confirm_url}")
                
                # If we found a token via extra patterns, try it
                if confirm_url:
                    headers = {"Referer": "https://drive.google.com", "User-Agent": self.session.headers.get("User-Agent")}
                    confirmed = self.session.get(confirm_url, timeout=60, stream=True, headers=headers)
                    if confirmed.status_code == 200:
                        # Use atomic write
                        try:
                            atomic_write_stream(confirmed, out_file, chunk_size=8192)
                        except Exception as e:
                            logger.debug(f"Atomic write failed: {e}")
                            return False
                        
                        if self.finalize_download(out_file, source_name):
                            return True
                        else:
                            self.mark_skip_gdrive(file_id, reason="invalid_after_extra_token", hours=24)
                            return False

                # 3) JSON-ish or JS variables that embed downloadUrl / confirm_token
                m3 = re.search(r'["\']downloadUrl["\']\s*:\s*["\'](\/uc\?export=download[^"\']+)', body, flags=re.I)
                if m3:
                    confirm_path = urllib.parse.unquote(m3.group(1))
                    confirm_url = urljoin("https://drive.google.com", confirm_path)
                    logger.info(f"🔎 Found JS downloadUrl -> {confirm_url}")
                    headers = {"Referer": "https://drive.google.com", "User-Agent": self.session.headers.get("User-Agent")}
                    resp2 = self.session.get(confirm_url, timeout=60, stream=True, headers=headers)
                    if resp2.status_code == 200:
                        # Use atomic write
                        try:
                            atomic_write_stream(resp2, out_file, chunk_size=8192)
                        except Exception as e:
                            logger.debug(f"Atomic write failed: {e}")
                            return False
                        
                        if self.finalize_download(out_file, source_name):
                            return True
                        else:
                            self.mark_skip_gdrive(file_id, reason="invalid_after_js", hours=24)
                            return False

                # 4) cookie token (existing)
                cookie_token = None
                for cookie in self.session.cookies:
                    if 'download_warning' in cookie.name:
                        cookie_token = cookie.value
                        break
                if cookie_token:
                    final = f"{base}?id={file_id}&export=download&confirm={cookie_token}"
                    if resource_key:
                        final += f"&resourcekey={urllib.parse.quote_plus(resource_key)}"
                    logger.info(f"🔎 Found download_warning cookie -> trying {final}")
                    headers = {"Referer": "https://drive.google.com", "User-Agent": self.session.headers.get("User-Agent")}
                    final_resp = self.session.get(final, timeout=60, stream=True, headers=headers)
                    if final_resp.status_code == 200:
                        # Use atomic write
                        try:
                            atomic_write_stream(final_resp, out_file, chunk_size=8192)
                        except Exception as e:
                            logger.debug(f"Atomic write failed: {e}")
                            return False
                        
                        if self.finalize_download(out_file, source_name):
                            return True
                        else:
                            self.mark_skip_gdrive(file_id, reason="invalid_after_cookie", hours=24)
                            return False

                # 5) fallback confirm=1
                try:
                    final_try = f"{base}?id={file_id}&export=download&confirm=1"
                    if resource_key:
                        final_try += f"&resourcekey={urllib.parse.quote_plus(resource_key)}"
                    headers = { "Referer": "https://drive.google.com", "User-Agent": self.session.headers.get("User-Agent") }
                    logger.info(f"🔎 Trying fallback confirm=1 -> {final_try}")
                    final_resp = self.session.get(final_try, timeout=60, stream=True, allow_redirects=True, headers=headers)
                    if final_resp.status_code == 200:
                        # Use atomic write
                        try:
                            atomic_write_stream(final_resp, out_file, chunk_size=8192)
                        except Exception as e:
                            logger.debug(f"Atomic write failed: {e}")
                            return False
                        
                        if self.finalize_download(out_file, source_name):
                            return True
                        else:
                            self.mark_skip_gdrive(file_id, reason="invalid_after_confirm1", hours=24)
                            return False
                except Exception as e:
                    logger.debug(f"final fallback failed: {e}")

                # If none of the above succeeded, attempt Selenium-assisted click on 'download anyway' button
                try:
                    logger.info("🔁 Attempting Selenium-assisted GDrive fallback (click 'Download anyway' if present)")
                    if self.download_gdrive_file_via_selenium(file_id, source_name, resource_key=resource_key):
                        # validate file
                        return True
                    else:
                        self.mark_skip_gdrive(file_id, reason="invalid_after_selenium", hours=24)
                        return False
                except Exception as e:
                    logger.debug(f"Selenium-assisted fallback error: {e}")

                # If we reach here, we couldn't get a valid PDF; mark skip for 24h
                logger.error(f"❌ GDrive download failed: {resp.status_code} (no PDF found / confirm token missing)")
                self.mark_skip_gdrive(file_id, reason="no_confirm_token", hours=24)
                return False

            except Exception as e:
                logger.error(f"❌ Error downloading from Google Drive: {e}")
                try:
                    out_file.unlink(missing_ok=True)
                except:
                    pass
                return False

    def download_direct_file(self, url, source_name, _depth=0):
        """Download a direct PDF file with robust fallbacks.
        Adds recursion depth to avoid infinite loops when following viewer pages.
        """

        # ====== PATCH ======
        # If the URL looks like a Google Drive file/viewer, route to the gdrive downloader
        gdrive_id = self.extract_gdrive_file_id(url)
        if gdrive_id:
            resource_key = self.extract_gdrive_resource_key(url)
            logger.info(f"🔁 Detected Google Drive link, routing to download_gdrive_file for {gdrive_id} (resource_key={resource_key})")
            # call the gdrive-specific downloader which handles confirmation redirects/scanned pages
            return self.download_gdrive_file(gdrive_id, source_name, resource_key=resource_key)
        # ====== END PATCH ======

        with self.download_lock:
            # Check if this URL is already being downloaded in another thread
            if url in self._in_progress:
                logger.info(f"⏸️ Download already in progress for {url}, skipping")
                return False
            self._in_progress.add(url)

        download_dir = Path(self.config["settings"]["download_dir"])
        today_str = datetime.now().strftime("%Y%m%d")
        safe_name = sanitize_filename(f"{source_name}_{today_str}.pdf")
        out_file = download_dir / safe_name

        try:
            # --- PATCH 5: Enhanced recursion depth logging ---
            if _depth > self.MAX_DOWNLOAD_DEPTH:
                logger.error(f"❌ Max recursion depth reached for {url}")
                logger.error(f"❌ Blocked infinite loop on URL: {url}")
                return False

            logger.info(f"⬇️ Download attempt (depth={_depth}): {url}")
            # allow redirects on GET to reach direct pdf if server redirects
            response = self.session.get(url, timeout=60, stream=True, allow_redirects=True)

            if response.status_code != 200:
                logger.warning(f"⚠️ Direct download failed: {response.status_code} for {url}")
                return False

            content_type = response.headers.get('content-type', '').lower()
            content_length = response.headers.get('content-length')

            # If it's already a PDF-like response, save normally
            if 'pdf' in content_type or (content_length and int(content_length) > self.MIN_ACCEPTED_FILESIZE) or response.url.lower().endswith('.pdf'):
                # Use atomic write
                try:
                    atomic_write_stream(response, out_file, chunk_size=8192)
                except Exception as e:
                    logger.debug(f"Atomic write failed: {e}")
                    return False
                
                # Use finalize_download for validation and bookkeeping
                return self.finalize_download(out_file, source_name)

            # ---------- FALLBACK: parse HTML for candidate PDF/Drive links ----------
            body = response.text or ""
            
            # Detect "No preview available / can't preview" pages and force the GDrive download flow
            html_lower = body.lower()
            if "can't preview file" in html_lower or "no preview available" in html_lower or "google drive - can't open file" in html_lower:
                # try to extract file id and route to gdrive downloader
                fid = self.extract_gdrive_file_id(response.url)
                if fid:
                    logger.info(f"🔁 Detected Drive preview page without preview; routing to GDrive download for {fid}")
                    resource_key = self.extract_gdrive_resource_key(response.url)
                    return self.download_gdrive_file(fid, source_name, resource_key=resource_key)

            # 1) quick regex for any absolute .pdf in the HTML (most common)
            m_pdf = re.search(r"https?://[^\s'\"<>]+\.pdf", body, flags=re.I)
            if m_pdf:
                pdf_url = m_pdf.group(0)
                logger.info(f"🔎 Found .pdf URL in HTML; following: {pdf_url}")
                return self.download_direct_file(pdf_url, source_name, _depth=_depth+1)

            # 2) check common tag attributes: anchors, iframes, embeds
            soup = BeautifulSoup(body, "html.parser")

            # anchors
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                resolved = urljoin(response.url, href)
                # if pdf-like, follow
                if resolved.lower().endswith(".pdf") or ".pdf" in resolved.lower():
                    logger.info(f"🔎 Found PDF via anchor: {resolved}")
                    return self.download_direct_file(resolved, source_name, _depth=_depth+1)
                # try resolving viewer/drive patterns
                resolved_candidate = self.try_resolve_viewer_link(href) or self.try_resolve_viewer_link(resolved)
                if resolved_candidate:
                    logger.info(f"🔎 Resolved viewer link from anchor: {resolved_candidate}")
                    return self.download_direct_file(resolved_candidate, source_name, _depth=_depth+1)

            # iframes / embed / object tags
            for tag in soup.find_all(['iframe', 'embed', 'object']):
                src = tag.get('src') or tag.get('data-src') or tag.get('data') or tag.get('data-original')
                if not src:
                    continue
                resolved = urljoin(response.url, src.strip())
                if resolved.lower().endswith(".pdf") or ".pdf" in resolved.lower():
                    logger.info(f"🔎 Found PDF via iframe/embed: {resolved}")
                    return self.download_direct_file(resolved, source_name, _depth=_depth+1)
                resolved_candidate = self.try_resolve_viewer_link(resolved) or self.try_resolve_viewer_link(src)
                if resolved_candidate:
                    logger.info(f"🔎 Resolved viewer link from iframe/embed: {resolved_candidate}")
                    return self.download_direct_file(resolved_candidate, source_name, _depth=_depth+1)

            # 3) search inline scripts for .pdf or drive links
            for script in soup.find_all("script"):
                text = script.string or ""
                m_pdf = re.search(r"https?://[^\s'\"<>]+\.pdf", text, flags=re.I)
                if m_pdf:
                    pdf_url = m_pdf.group(0)
                    logger.info(f"🔎 Found .pdf URL inside inline script; following: {pdf_url}")
                    return self.download_direct_file(pdf_url, source_name, _depth=_depth+1)
                resolved_candidate = self.try_resolve_viewer_link(text)
                if resolved_candidate:
                    logger.info(f"🔎 Resolved viewer link inside script: {resolved_candidate}")
                    return self.download_direct_file(resolved_candidate, source_name, _depth=_depth+1)

            # 4) HEAD fallback: sometimes server exposes pdf via HEAD redirect
            try:
                head = self.session.head(url, allow_redirects=True, timeout=15)
                hct = head.headers.get('content-type', '').lower()
                if 'pdf' in hct or head.url.lower().endswith('.pdf'):
                    logger.info(f"🔎 HEAD suggests PDF at {head.url}")
                    return self.download_direct_file(head.url, source_name, _depth=_depth+1)
            except Exception as e:
                logger.debug(f"HEAD fallback failed: {e}")

            logger.warning(f"⚠️ Response doesn't appear to be PDF (content-type={content_type}) and no candidate link found for {url}")
            return False

        except Exception as e:
            logger.error(f"❌ Error in direct download: {e}")
            return False
        finally:
            with self.download_lock:
                self._in_progress.discard(url)

    def download_telegram_sources(self):
        """Download from Telegram sources using telethon"""
        # Check if Telegram is enabled in config
        if not self.config["settings"].get("enable_telegram", True):
            logger.info("ℹ️ Telegram downloads disabled by config")
            return []
            
        telegram_sources = self.config.get("telegram_sources", {})
        if not telegram_sources:
            logger.info("ℹ️ No Telegram sources configured")
            return []
        
        downloaded_files = []
        
        for source_name, source_config in telegram_sources.items():
            try:
                logger.info(f"📱 Processing Telegram source: {source_name}")
                
                # Check if telethon is available
                if TelegramClient is None:
                    logger.error("❌ Telethon not installed. Install with: pip install telethon")
                    continue
                
                papers_config = source_config.get("newspapers", [])
                grouped_papers = {}

                # If dict → proper grouping
                if isinstance(papers_config, dict):
                    grouped_papers = {group: list(set(variants)) for group, variants in papers_config.items()}
                else:
                    # backward support, treat as one group
                    grouped_papers = {"default": list(set(papers_config))}

                today_str = datetime.now().strftime("%Y%m%d")
                telegram_files_to_download = []

                # Build download list group-wise (only one file per group allowed)
                for group, variants in grouped_papers.items():
                    group_already_done = False
                    
                    for variant in variants:
                        filename = sanitize_filename(f"telegram_{clean_name(group)}_{today_str}.pdf")
                        file_path = Path(self.config["settings"]["download_dir"]) / filename

                        # Already downloaded any city in this group?
                        if file_path.exists() and file_path.stat().st_size > self.MIN_ACCEPTED_FILESIZE and self.validate_pdf_file(file_path):
                            status(f"✅ Already downloaded {group}, skipping...")
                            logger.info(f"✅ Already downloaded {group} (any city)")
                            group_already_done = True
                            break
                    
                    if not group_already_done:
                        telegram_files_to_download.append((group, variants))

                if not telegram_files_to_download:
                    logger.info(f"✅ All Telegram newspapers already downloaded for {source_name}")
                    continue
                
                # Download from Telegram
                # keep a snapshot of requested groups so we can find which were actually downloaded
                requested_groups = [g for g, _ in telegram_files_to_download]
                success = self.download_from_telegram(
                    source_config,
                    telegram_files_to_download,
                    source_name
                )

                if success:
                    # telegram_files_to_download is mutated by download_from_telegram (removed groups)
                    remaining = [g for g, _ in telegram_files_to_download]
                    downloaded_now = [g for g in requested_groups if g not in remaining]
                    downloaded_files.extend(downloaded_now)
                    self.run_downloaded_files.extend([sanitize_filename(f"telegram_{clean_name(g)}_{today_str}.pdf") for g in downloaded_now])
                    
            except Exception as e:
                logger.error(f"❌ Error processing Telegram source {source_name}: {e}")
        
        return downloaded_files

    def download_from_telegram(self, telegram_config, newspapers_to_download, source_name):
        """Download specific newspapers from Telegram channel"""
        try:
            if TelegramClient is None:
                logger.error("❌ Telethon not installed. Install with: pip install telethon")
                return False
            
            api_id = telegram_config.get("api_id")
            api_hash = telegram_config.get("api_hash")
            channel_id = telegram_config.get("channel_id")
            channel_username = telegram_config.get("channel_username")
            days_back = telegram_config.get("days_back", 1)
            
            if not api_id or not api_hash:
                logger.error("❌ Telegram API credentials missing")
                return False
            
            logger.info(f"🔍 Searching Telegram channel: {channel_id or channel_username}")
            logger.info(f"📰 Looking for newspaper groups: {[group for group, _ in newspapers_to_download]}")
            
            # Run async function in event loop
            async def telegram_download_async():
                client = None
                try:
                    client = TelegramClient(f'telegram_session_{source_name}', api_id, api_hash)
                    await client.start()
                    
                    # Get the channel entity - support both ID and username
                    try:
                        if channel_id:
                            channel = await client.get_entity(channel_id)
                        else:
                            channel = await client.get_entity(channel_username)
                    except Exception as e:
                        logger.error(f"❌ Could not access Telegram channel {channel_id or channel_username}: {e}")
                        return False
                    
                    # Calculate date range
                    end_date = datetime.now(timezone.utc)
                    start_date = end_date - timedelta(days=days_back)

                    downloaded_count = 0
                    
                    # Search for messages with the specific newspapers
                    async for message in client.iter_messages(channel, limit=100):
                        msg_date = message.date

                        # Convert message.date to UTC if naive
                        if msg_date.tzinfo is None:
                            msg_date = msg_date.replace(tzinfo=timezone.utc)

                        if msg_date < start_date:
                            break
                            
                        if message.media and isinstance(message.media, MessageMediaDocument):
                            # Check if it's a PDF
                            if (hasattr(message.media.document, 'mime_type') and 
                                message.media.document.mime_type == 'application/pdf'):
                                
                                # Check filename patterns for our specific newspapers
                                for attr in message.media.document.attributes:
                                    if hasattr(attr, 'file_name'):
                                        filename = attr.file_name
                                        
                                        # newspapers_to_download holds tuples: (group, [variants])
                                        for group, variants in list(newspapers_to_download):
                                            for variant in variants:
                                                if fuzzy_match(variant, filename):

                                                    today_str = datetime.now().strftime("%Y%m%d")
                                                    out_filename = sanitize_filename(f"telegram_{clean_name(group)}_{today_str}.pdf")
                                                    out_file = Path(self.config["settings"]["download_dir"]) / out_filename

                                                    await message.download_media(file=out_file)
                                                    
                                                    # Use finalize_download for Telegram files too
                                                    source_name_for_set = f"telegram_{clean_name(group)}"
                                                    if self.finalize_download(out_file, source_name_for_set):
                                                        status(f"✅ Downloaded {group} from Telegram")
                                                        logger.info(f"✅ Downloaded {group} ({variant}) from Telegram")
                                                        newspapers_to_download.remove((group, variants))   # stop group
                                                        downloaded_count += 1
                                                        break
                                                    else:
                                                        logger.warning(f"⚠️ Telegram-supplied PDF invalid or too small: {out_file}")
                                        
                                        # Break if we found all newspapers
                                        if not newspapers_to_download:
                                            break
                    
                    logger.info(f"✅ Downloaded {downloaded_count} newspaper groups from Telegram")
                    return downloaded_count > 0
                    
                except Exception as e:
                    logger.error(f"❌ Error downloading from Telegram: {e}")
                    return False
                finally:
                    if client:
                        await client.disconnect()
            
            # Increased Telegram timeout and better error handling with asyncio loop detection
            # give Telegram more time for slow channels (180s)
            try:
                return asyncio.run(asyncio.wait_for(telegram_download_async(), timeout=180))
            except RuntimeError:
                # running inside existing loop (e.g., in Jupyter). Use create_task / run_until_complete fallback.
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(asyncio.wait_for(telegram_download_async(), timeout=180))
            except Exception as e:
                logger.error(f"❌ Telegram fetch error/timeout: {e}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error in Telegram download: {e}")
            return False

    def download_all_optimized(self):
        """Main optimized download function"""
        status("📥 Downloading today's newspapers...")
        
        logger.info("=" * 60)
        logger.info("🗞️  COMPLETE NEWSPAPER DOWNLOADER")
        logger.info(f"   Batch Size: {self.batch_size}")
        logger.info(f"   Preprocess Workers: {self.max_preprocess_workers}")
        logger.info(f"   Already Downloaded: {len(self.already_downloaded)} files")
        logger.info(f"   Web Sources: {len(self.web_sources)}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Phase 1: Parallel preprocessing
        logger.info("\n🚀 PHASE 1: Parallel Preprocessing")
        preprocessed_sources = self.preprocess_all_sources_parallel()
        
        # Separate results
        successful_preprocess = [item for item in preprocessed_sources if item["success"]]
        failed_preprocess = [item for item in preprocessed_sources if not item["success"]]
        
        new_downloads = [item for item in successful_preprocess if not item.get("already_downloaded", False)]
        already_downloaded = [item for item in successful_preprocess if item.get("already_downloaded", False)]
        
        logger.info(f"✅ New downloads to process: {len(new_downloads)}")
        logger.info(f"✅ Already downloaded: {len(already_downloaded)}")
        logger.info(f"❌ Failed preprocessing: {len(failed_preprocess)}")
        
        # Phase 2: Batched serial downloads
        logger.info("\n🚀 PHASE 2: Batched Serial Downloads")
        
        total_downloaded = len(self.already_downloaded)
        
        if new_downloads:
            batches = []
            for i in range(0, len(new_downloads), self.batch_size):
                batch = new_downloads[i:i + self.batch_size]
                batches.append(batch)
            
            logger.info(f"📦 Created {len(batches)} batch(es) for download")
            
            for i, batch in enumerate(batches, 1):
                logger.info(f"\n📦 Processing batch {i}/{len(batches)} ({len(batch)} items)")
                batch_start = time.time()
                
                successful_downloads = self.download_batch_serial(batch)
                total_downloaded += len(successful_downloads)
                
                batch_time = time.time() - batch_start
                logger.info(f"⏱️  Batch {i} completed in {batch_time:.1f}s - {len(successful_downloads)}/{len(batch)} successful")
        else:
            logger.info("ℹ️ No new downloads to process")
        
        # Phase 3: Telegram downloads (only if enabled)
        logger.info("\n🚀 PHASE 3: Telegram Downloads")
        if not self.config["settings"].get("enable_telegram", True):
            logger.info("ℹ️ Telegram downloads disabled by config")
            telegram_downloads = []
        else:
            telegram_downloads = self.download_telegram_sources()
        
        total_downloaded += len(telegram_downloads)
        
        # Summary
        total_time = time.time() - start_time
        
        # Get all downloaded files in directory
        download_dir = Path(self.config["settings"]["download_dir"])
        downloaded_files = [p.name for p in download_dir.glob("*.pdf") if self.validate_pdf_file(p)]
        
        # Terminal summary
        print("\n")  # Clear the status line
        print("📦 Newspaper Fetch Summary:")
        print(f"  ✅ Files now in download_dir: {len(downloaded_files)}")
        for fn in sorted(downloaded_files):
            print(f"    - {fn}")
        
        # Show exact PDF download results in summary
        print("\n📄 Fresh downloads (this run):")
        if self.run_downloaded_files:
            for fn in sorted(self.run_downloaded_files):
                print(f"    - {fn}")
        else:
            print("    (None)")
        
        print(f"\n  ❗ Preprocess failures: {len(failed_preprocess)}")
        for item in failed_preprocess:
            print(f"    - {item.get('source')}: {item.get('error', 'Unknown error')}")
        print(f"\n  📲 Telegram (successful groups): {len(telegram_downloads)}")
        for t in telegram_downloads:
            print(f"    - {t}")
        print("\n📝 Full details in newspaper_downloader.log\n")
        
        # Log file summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 COMPLETE DOWNLOAD SUMMARY")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total time: {total_time:.1f}s")
        logger.info(f"✅ Total files available: {total_downloaded}")
        logger.info(f"   - Web/Direct downloads: {len(self.already_downloaded) + (total_downloaded - len(self.already_downloaded) - len(telegram_downloads))}")
        logger.info(f"   - Telegram downloads: {len(telegram_downloads)}")
        logger.info(f"❌ Failed preprocessing: {len(failed_preprocess)}")
        
        if failed_preprocess:
            logger.info("\n❌ Failed preprocesses:")
            for item in failed_preprocess:
                logger.info(f"   - {item['source']}: {item.get('error', 'Unknown error')}")
        
        if telegram_downloads:
            logger.info("\n✅ Telegram downloads:")
            for newspaper in telegram_downloads:
                logger.info(f"   - {newspaper}")

        return total_downloaded > 0

def main():
    """Main execution function"""
    try:
        downloader = CompleteNewspaperDownloader(batch_size=4, max_preprocess_workers=1)  # Reduced to 1 for debugging
        success = downloader.download_all_optimized()
        return success
        
    except Exception as e:
        logger.error(f"💥 Critical error: {e}")
        return False
        
    finally:
        if 'downloader' in locals():
            downloader.close_drivers()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)