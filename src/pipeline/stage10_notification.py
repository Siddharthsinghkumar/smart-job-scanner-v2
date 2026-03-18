#!/usr/bin/env python3
"""
10_notify_shortlist_telegram.py – Telegram notifier for newly shortlisted jobs
"""

import os
import re
import sqlite3
import requests
import logging
import time
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import html

# ─── Load environment variables ───
load_dotenv()

# ─── Configuration ───
DEBUG_MODE = os.getenv("DEBUG", "0") == "1"

# Hardcoded paths
HISTORY_DB = Path("data/shortlist_history.db")
STATE_FILE = Path("data/telegram/last_telegram_sent.txt")
LOG_DIR = Path("logs")
DATA_DIR = Path("data")

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ─── Validation ───
if not BOT_TOKEN:
    raise SystemExit("❌ TELEGRAM_BOT_TOKEN not found in environment variables")
if not CHAT_ID:
    raise SystemExit("❌ TELEGRAM_CHAT_ID not found in environment variables")

# ─── Pre-compiled regex patterns ───
# Improved employer pattern to catch organization names
EMPLOYER_PATTERN = re.compile(r"([A-Z][A-Za-z\s&]+(?:Limited|Corporation|Institute|University|Commission|Company|Services|Solutions|Power|Authority|Board|Department))", re.IGNORECASE)
DATE_PATTERN = re.compile(r"(?:apply|last\s*date|by)\s*(\d{1,2}\s*\w+\s*\d{4})", re.IGNORECASE)
WEBSITE_PATTERN = re.compile(r"(https?://[^\s]+)")

# ─── Argument Parsing ───
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Send Telegram job notifications')
    parser.add_argument('--force', action='store_true', 
                       help='Force send all jobs, ignore state file')
    parser.add_argument('--since', type=str,
                       help='Send jobs since specific date (YYYY-MM-DD)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be sent without actually sending')
    return parser.parse_args()

# ─── Logging Setup ───
def setup_logging():
    """Configure comprehensive logging to file and console"""
    log_filename = LOG_DIR / f"telegram_notify_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.DEBUG if DEBUG_MODE else logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Create logger instance
    logger = logging.getLogger(__name__)
    logger.info(f"🔧 Starting Telegram Notifier")
    logger.info(f"📂 Log file: {log_filename}")
    logger.info(f"📊 History DB: {HISTORY_DB}")
    logger.info(f"🕓 State file: {STATE_FILE}")
    logger.info(f"🔍 Debug mode: {DEBUG_MODE}")
    
    return logger

# Initialize logging
logger = setup_logging()

# ─── HTML Escaping for Telegram ───
def escape_html(text: str) -> str:
    """Escape text for HTML parse_mode in Telegram"""
    if not text:
        return ""
    return html.escape(str(text))

def escape_html_codeblock(text: str) -> str:
    """Escape special characters for Telegram HTML parse_mode (inside <code> or <pre> blocks)."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("-", "&#45;")
            .replace(".", "&#46;")
            .replace("(", "&#40;")
            .replace(")", "&#41;")
            .replace("+", "&#43;")
            .replace("=", "&#61;")
            .replace("|", "&#124;")
    )

# ─── Telegram Bot Validation ───
def validate_telegram_config() -> bool:
    """
    Validate that bot token and chat ID are correctly configured
    Returns True if configuration is valid, False otherwise
    """
    logger.info("🔍 Validating Telegram configuration...")
    
    # Test bot token by getting bot info
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            bot_info = response.json()
            if bot_info.get('ok'):
                bot_username = bot_info['result']['username']
                logger.info(f"✅ Bot validated: @{bot_username}")
                
                # Test sending a message to validate chat ID - use PLAIN TEXT for validation
                test_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
                test_payload = {
                    "chat_id": CHAT_ID,
                    "text": "🔧 Bot configuration test - please ignore",  # Plain text, no formatting
                    # No parse_mode for validation - use plain text to avoid formatting issues
                }
                
                test_response = requests.post(test_url, json=test_payload, timeout=10)
                if test_response.status_code == 200:
                    logger.info("✅ Chat ID validated successfully")
                    return True
                else:
                    error_info = test_response.json()
                    error_description = error_info.get('description', 'Unknown error')
                    logger.error(f"❌ Chat ID validation failed: {error_description}")
                    
                    if "bot can't send messages to bots" in error_description.lower():
                        logger.error("💡 SOLUTION: Your TELEGRAM_CHAT_ID is set to a BOT ID, not a USER ID!")
                        logger.error("💡 To get your correct USER Chat ID:")
                        logger.error("   1. Start a chat with @userinfobot on Telegram")
                        logger.error("   2. It will immediately reply with your Chat ID")
                        logger.error("   3. Use that number in your .env file as TELEGRAM_CHAT_ID")
                        logger.error("💡 Your Chat ID should be a number like: 123456789")
                    elif "chat not found" in error_description.lower():
                        logger.error("💡 SOLUTION: Chat ID is invalid or you haven't started a chat with the bot")
                        logger.error("💡 Send a '/start' message to your bot first")
                    elif "forbidden" in error_description.lower():
                        logger.error("💡 SOLUTION: Bot is blocked by user or doesn't have permission to send messages")
                    else:
                        logger.error("💡 Check your TELEGRAM_CHAT_ID configuration")
                    
                    return False
            else:
                logger.error("❌ Invalid bot token")
                logger.error("💡 Check your TELEGRAM_BOT_TOKEN in .env file")
                return False
        else:
            logger.error(f"❌ Failed to validate bot token: {response.text}")
            logger.error("💡 Check your TELEGRAM_BOT_TOKEN in .env file")
            return False
    except Exception as e:
        logger.error(f"❌ Error validating Telegram configuration: {e}")
        return False

# ─── Telegram Message Sender with Retries ───
def send_telegram_message(text: str, max_retries: int = 3, initial_delay: float = 3.0) -> bool:
    """Send message to Telegram bot with exponential backoff retry mechanism"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID, 
        "text": text, 
        "parse_mode": "HTML",  # Using HTML for more reliable formatting
        "disable_web_page_preview": True
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=15)
            if response.status_code == 200:
                logger.debug("✅ Telegram message sent successfully")
                return True
            else:
                error_info = response.json()
                error_description = error_info.get('description', 'Unknown error')
                logger.warning(f"⚠️ Telegram API error (attempt {attempt+1}/{max_retries}): {response.status_code} - {error_description}")
        except requests.exceptions.Timeout:
            logger.warning(f"⚠️ Telegram timeout (attempt {attempt+1}/{max_retries})")
        except requests.exceptions.ConnectionError:
            logger.warning(f"⚠️ Telegram connection error (attempt {attempt+1}/{max_retries})")
        except Exception as e:
            logger.error(f"❌ Unexpected Telegram error (attempt {attempt+1}/{max_retries}): {e}")
        
        # Exponential backoff before retry
        if attempt < max_retries - 1:
            delay = initial_delay * (2 ** attempt)
            logger.info(f"⏳ Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
    
    logger.error(f"❌ Failed to send Telegram message after {max_retries} attempts")
    return False

# ─── State Management ───
def get_last_sent_time() -> datetime:
    """Get the timestamp of last successful notification send"""
    if not STATE_FILE.exists():
        logger.info("🆕 No previous state file found, starting from beginning")
        return datetime.fromtimestamp(0)
    
    try:
        last_time_str = STATE_FILE.read_text().strip()
        last_time = datetime.fromisoformat(last_time_str)
        logger.info(f"🕓 Last notification sent: {last_time}")
        return last_time
    except ValueError as e:
        logger.warning(f"⚠️ Corrupted state file, resetting: {e}")
        return datetime.fromtimestamp(0)
    except Exception as e:
        logger.error(f"❌ Error reading state file: {e}")
        return datetime.fromtimestamp(0)

def update_last_sent_time(timestamp: datetime = None):
    """Update the state file with specified timestamp"""
    try:
        target_time = timestamp or datetime.now()
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        STATE_FILE.write_text(target_time.isoformat())
        logger.info(f"✅ Updated last sent timestamp → {target_time}")
    except Exception as e:
        logger.error(f"❌ Failed to update state file: {e}")

# ─── Database Operations ───
def get_new_jobs(last_sent_time: datetime) -> list:
    """Fetch jobs shortlisted after the last sent time"""
    if not HISTORY_DB.exists():
        logger.error(f"❌ History database not found: {HISTORY_DB}")
        return []
    
    try:
        conn = sqlite3.connect(HISTORY_DB)
        cur = conn.cursor()
        
        # Verify table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sent_jobs'")
        if not cur.fetchone():
            logger.error("❌ 'sent_jobs' table not found in database")
            conn.close()
            return []
        
        # Get new jobs using direct timestamp comparison
        query = """
            SELECT job_id, job_text, similarity, match_score, processed_at, shortlisted_at
            FROM sent_jobs
            WHERE shortlisted_at > ?
            ORDER BY shortlisted_at ASC
        """
        last_sent_iso = last_sent_time.isoformat()
        cur.execute(query, (last_sent_iso,))
        rows = cur.fetchall()
        conn.close()
        
        logger.info(f"📊 Found {len(rows)} new jobs since {last_sent_time}")
        return rows
        
    except sqlite3.Error as e:
        logger.error(f"❌ Database error: {e}")
        return []

def get_all_jobs() -> list:
    """Fetch all jobs from the database"""
    if not HISTORY_DB.exists():
        logger.error(f"❌ History database not found: {HISTORY_DB}")
        return []
    
    try:
        conn = sqlite3.connect(HISTORY_DB)
        cur = conn.cursor()
        
        # Verify table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sent_jobs'")
        if not cur.fetchone():
            logger.error("❌ 'sent_jobs' table not found in database")
            conn.close()
            return []
        
        # Get all jobs
        query = """
            SELECT job_id, job_text, similarity, match_score, processed_at, shortlisted_at
            FROM sent_jobs
            ORDER BY shortlisted_at ASC
        """
        cur.execute(query)
        rows = cur.fetchall()
        conn.close()
        
        logger.info(f"📊 Found {len(rows)} total jobs in database")
        return rows
        
    except sqlite3.Error as e:
        logger.error(f"❌ Database error: {e}")
        return []

# ─── Job Text Processing ───
def parse_job_text(text: str) -> dict:
    """Extract structured information from job text with improved employer detection"""
    try:
        # Try multiple patterns to extract employer
        employer = "Unknown Organization"
        
        # Pattern 1: Look for organization names with common suffixes
        employer_match = EMPLOYER_PATTERN.search(text)
        if employer_match:
            employer = employer_match.group(1).strip()
        else:
            # Pattern 2: Look for text before first pipe (|) and clean it
            lines = text.split('\n')
            if lines:
                first_line = lines[0]
                if '|' in first_line:
                    employer_candidate = first_line.split('|')[0].strip()
                    # Clean up numbering and dashes
                    employer_candidate = re.sub(r'^\d+\.\s*-\s*', '', employer_candidate)
                    if employer_candidate and len(employer_candidate) > 3:
                        employer = employer_candidate
        
        # Extract date with improved pattern
        date_match = DATE_PATTERN.search(text)
        apply_by = date_match.group(1) if date_match else "Not specified"
        
        # Extract website
        website_match = WEBSITE_PATTERN.search(text)
        website = website_match.group(1) if website_match else "No URL provided"
        
        return {
            "employer": employer,
            "apply_by": apply_by,
            "website": website
        }
    except Exception as e:
        logger.warning(f"⚠️ Error parsing job text: {e}")
        return {"employer": "Unknown Organization", "apply_by": "Not specified", "website": "No URL provided"}

def choose_resume(job_text: str) -> str:
    """Select appropriate resume based on job content"""
    text_lower = job_text.lower()
    
    if any(keyword in text_lower for keyword in ["engineer", "electrical", "transformer", "bms", "plc", "scada"]):
        return "dynamic_resume_ee.txt"
    elif any(keyword in text_lower for keyword in ["data", "ai", "ml", "python", "vision", "deep", "model"]):
        return "dynamic_resume_aiml.txt"
    else:
        return "dynamic_resume_hybrid.txt"

def suggest_improvement(job_text: str) -> str:
    """Generate improvement suggestions based on job requirements"""
    text_lower = job_text.lower()
    
    if "experience" in text_lower:
        return "Add recent project experience section to resume."
    elif "communication" in text_lower or "team" in text_lower:
        return "Highlight teamwork and presentation achievements."
    elif "portfolio" in text_lower or "github" in text_lower:
        return "Ensure portfolio and GitHub are updated with relevant projects."
    elif any(keyword in text_lower for keyword in ["scada", "transformer", "plc"]):
        return "Add relevant SCADA or Transformer project experience."
    elif any(keyword in text_lower for keyword in ["python", "machine learning", "ai"]):
        return "Highlight Python and ML projects in your portfolio."
    else:
        return "Tailor resume to match specific keywords from job description."

def extract_apply_date(job_text: str) -> str:
    """Extract and format application deadline"""
    date_match = DATE_PATTERN.search(job_text)
    if date_match:
        return f"🗓 Apply before: {date_match.group(1)}"
    return "📅 Deadline: Not specified"

# ─── Message Formatting ───
def format_job_message(job_data: tuple) -> str:
    """Format a single job notification message with HTML formatting"""
    job_id, job_text, similarity, score, processed_at, shortlisted_at = job_data
    parsed = parse_job_text(job_text)
    resume = choose_resume(job_text)
    tip = suggest_improvement(job_text)
    apply_date = extract_apply_date(job_text)
    
    # Escape ALL text for HTML
    safe_employer = escape_html(parsed['employer'])
    safe_text = escape_html_codeblock(job_text[:160] + "..." if len(job_text) > 160 else job_text)
    safe_apply_by = escape_html(parsed['apply_by'])
    safe_tip = escape_html(tip)
    safe_resume = escape_html_codeblock(resume)
    dt = datetime.fromisoformat(shortlisted_at)
    safe_shortlisted = escape_html(f"{dt.day} {dt.strftime('%B %Y')}")
    
    # Handle None score
    display_score = score if score is not None else "N/A"
    
    # Format message with HTML syntax
    message = (
        f"<b>📢 New Job Alert</b>\n"
        f"<b>🏢 {safe_employer}</b>\n"
        f"<code>{safe_text}</code>\n"
        f"<b>{apply_date}</b>\n"
        f"<b>🌐</b> {parsed['website']}\n"
        f"<b>🎯 Similarity:</b> {similarity:.2f} | <b>Score:</b> {display_score}/5\n"
        f"<b>📄 Resume:</b> <code>{safe_resume}</code>\n"
        f"<b>💼 Resume tip:</b> {safe_tip}\n"
        f"<i>Shortlisted on {safe_shortlisted}</i>"
    )
    
    return message

def format_summary_message(new_jobs: list, total_processed: int, successful_sends: int, failed_sends: int, runtime: str) -> str:
    """Format daily summary message with HTML formatting"""
    if not new_jobs:
        return ""
    
    latest_job_text = new_jobs[0][1]
    preview = escape_html_codeblock(latest_job_text[:80] + "..." if len(latest_job_text) > 80 else latest_job_text)
    current_time = escape_html(datetime.now().strftime('%d %b %Y, %H:%M'))
    
    summary = (
        f"<b>📊 Daily Summary</b>\n"
        f"• <b>Jobs scanned:</b> {total_processed}\n"
        f"• <b>Sent:</b> {successful_sends} new\n"
        f"• <b>Failed:</b> {failed_sends}\n"
        f"• <b>Runtime:</b> {runtime}\n"
        f"• <b>Latest job:</b> {preview}\n"
        f"• <b>Notified at:</b> {current_time}"
    )
    
    return summary

# ─── Main Execution ───
def main():
    """Main notification workflow"""
    args = parse_arguments()
    start_time = datetime.now()
    logger.info("🚀 Starting Telegram notification process")
    
    # Validate Telegram configuration first
    if not validate_telegram_config():
        logger.error("❌ Telegram configuration invalid. Exiting.")
        raise SystemExit(1)
    
    # Choose mode
    if args.force:
        logger.warning("⚠️ Force mode enabled — ignoring state file, sending all jobs.")
        jobs = get_all_jobs()
    elif args.since:
        try:
            since_time = datetime.fromisoformat(args.since)
            logger.info(f"📅 Sending jobs since {since_time}")
            jobs = get_new_jobs(since_time)
        except ValueError:
            logger.error("❌ Invalid --since date format (expected YYYY-MM-DD)")
            raise SystemExit(1)
    else:
        last_sent_time = get_last_sent_time()
        jobs = get_new_jobs(last_sent_time)
    
    if not jobs:
        logger.info("⚠️ No new jobs found to send.")
        raise SystemExit(0)

    logger.info(f"📤 Sending {len(jobs)} job notifications...")

    successful_sends = 0
    failed_sends = 0
    
    for job in jobs:
        job_id, job_text, similarity, score, _, shortlisted_at = job
        message = format_job_message(job)
        parsed = parse_job_text(job_text)
        
        if args.dry_run:
            print("\n--- DRY RUN MESSAGE ---\n", message)
            successful_sends += 1  # Count dry run as successful for logging
        else:
            if send_telegram_message(message):
                successful_sends += 1
                logger.info(f"✅ Sent job: {parsed['employer']} (similarity: {similarity:.2f})")
            else:
                failed_sends += 1
                logger.error(f"❌ Failed to send job: {parsed['employer']}")
            time.sleep(1.5)  # small delay between sends
    
    # Calculate runtime
    end_time = datetime.now()
    runtime_delta = end_time - start_time
    runtime_str = str(runtime_delta).split('.')[0]  # Remove microseconds
    
    if not args.dry_run and successful_sends > 0:
        # Update last sent timestamp to the latest job time
        last_timestamp = jobs[-1][5]  # shortlisted_at
        update_last_sent_time(datetime.fromisoformat(last_timestamp))
        
        # Send summary message
        summary_message = format_summary_message(jobs, len(jobs), successful_sends, failed_sends, runtime_str)
        send_telegram_message(summary_message)
        
        # Enhanced logging summary
        logger.info(f"📊 SUMMARY: Jobs scanned: {len(jobs)}, Sent: {successful_sends}, Failed: {failed_sends}, Runtime: {runtime_str}")
        print(f"📊 SUMMARY: Jobs scanned: {len(jobs)}, Sent: {successful_sends}, Failed: {failed_sends}, Runtime: {runtime_str}")
        
    elif args.dry_run:
        logger.info("🧪 Dry run completed (no messages actually sent).")
        print(f"🧪 Dry run completed: Would send {len(jobs)} messages")
    else:
        logger.error("❌ No notifications were sent successfully.")
        print("❌ No notifications were sent successfully.")

# ─── Entry Point ───
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("⏹️ Process interrupted by user")
        print("\n⏹️ Process interrupted by user")
    except Exception as e:
        logger.exception(f"💥 Fatal error in main execution: {e}")
        print(f"❌ Fatal error: {e}")
        raise