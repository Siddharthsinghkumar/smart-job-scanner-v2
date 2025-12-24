import json
import sqlite3
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# === LOGGING SETUP ===
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"smart_shortlist_{datetime.now():%Y_%m_%d_%H_%M}.log"

logging.basicConfig(
    filename=log_file,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(console)

logging.info("🚀 Smart Shortlist started")

# === PATHS & CONFIG ===
HISTORY_DB = Path("data/shortlist_history.db")
RESULTS_DIR = Path("data/llm_results")
SHORTLISTS_DIR = Path("data/shortlists")

# === HISTORY DATABASE SETUP ===
def init_db():
    """Initialize SQLite database for tracking already-processed jobs"""
    HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(HISTORY_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sent_jobs (
            job_id TEXT PRIMARY KEY,
            job_text TEXT,
            similarity REAL,
            match_score REAL,
            processed_at TEXT,
            shortlisted_at TEXT
        )
    """)
    conn.commit()
    return conn

def is_new_job(conn, job_id):
    """Check if job has been previously shortlisted"""
    c = conn.cursor()
    c.execute("SELECT 1 FROM sent_jobs WHERE job_id=?", (job_id,))
    return c.fetchone() is None

def log_job(conn, job):
    """Record a job in the history database"""
    try:
        conn.execute("""
            INSERT OR REPLACE INTO sent_jobs 
            (job_id, job_text, similarity, match_score, processed_at, shortlisted_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            job['job_id'], 
            job['job_text'],
            job['similarity'], 
            job['match_score_numeric'], 
            job['processed_at'],
            datetime.now().isoformat()
        ))
        conn.commit()
        logging.debug(f"✅ Logged job to history: {job['job_id']}")
    except Exception as e:
        logging.error(f"❌ Failed to log job to history: {e}")

def get_shortlist_history(conn, limit=10):
    """Get recent shortlist history for reference"""
    c = conn.cursor()
    c.execute("""
        SELECT job_id, job_text, similarity, match_score, shortlisted_at 
        FROM sent_jobs 
        ORDER BY shortlisted_at DESC 
        LIMIT ?
    """, (limit,))
    return c.fetchall()

# === MAIN SCRIPT ===
def smart_job_shortlist():
    """Create a truly useful job shortlist from LLM analyses with history tracking"""
    start_time = datetime.now()
    
    # Initialize database
    try:
        conn = init_db()
    except Exception as e:
        logging.error(f"❌ Database initialization failed: {e}")
        return
    
    # Find latest analysis
    latest_json = find_latest_json(RESULTS_DIR)
    
    if not latest_json:
        logging.error("❌ No analysis files found in %s", RESULTS_DIR)
        conn.close()
        return
    
    logging.info("📂 Analyzing: %s", latest_json.name)
    
    # Load data
    try:
        with open(latest_json, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"❌ Failed to load JSON data: {e}")
        conn.close()
        return
    
    # Smart filtering - look for actual positive signals
    all_promising_jobs = []
    
    for job in data:
        response = job.get("gemini_response", "").lower()
        
        # Multiple indicators of recommendation
        positive_indicators = [
            "recommended" in response and "not recommended" not in response,
            "match score: 4" in response or "match score: 5" in response,
            "match score: 3" in response and "recommended" in response,
            "verdict: recommended" in response
        ]
        
        if any(positive_indicators):
            # Extract key info reliably
            job_info = {
                'job_id': job['job_id'],
                'job_text': job['job_text'],
                'similarity': job['similarity'],
                'application_status': job.get('application_status', 'pending'),
                'processed_at': job['processed_at']
            }
            
            # Extract match score as both text and numeric
            match_score_text = "N/A"
            match_score_numeric = None
            
            if "match score:" in response:
                start = response.find("match score:") + len("match score:")
                end = response.find("/5", start)
                if end != -1:
                    match_score_text = response[start:end].strip()
                    # Convert to numeric if possible
                    try:
                        match_score_numeric = float(match_score_text)
                    except (ValueError, TypeError):
                        match_score_numeric = None
            
            job_info['match_score_text'] = match_score_text
            job_info['match_score_numeric'] = match_score_numeric
            all_promising_jobs.append(job_info)
    
    if not all_promising_jobs:
        logging.warning("⚠️ No promising jobs found with current criteria")
        
        # Show recent history for context
        history = get_shortlist_history(conn)
        if history:
            logging.info("📜 Recent shortlist history:")
            for job_id, job_text, similarity, match_score, shortlisted_at in history:
                score_display = f"{match_score}/5" if match_score else "N/A"
                logging.info("   - %s | %s: %s... (%s)", score_display, similarity, job_text[:60], shortlisted_at[:10])
        
        conn.close()
        return
    
    # Filter out already-seen jobs
    new_promising_jobs = []
    duplicate_count = 0
    
    for job in all_promising_jobs:
        if is_new_job(conn, job['job_id']):
            new_promising_jobs.append(job)
            log_job(conn, job)  # Log immediately to prevent future duplicates
        else:
            duplicate_count += 1
    
    logging.info("📊 Found %d promising jobs (%d duplicates, %d new)", 
                 len(all_promising_jobs), duplicate_count, len(new_promising_jobs))
    
    if not new_promising_jobs:
        logging.info("🎉 All promising jobs have already been shortlisted in previous runs!")
        
        # Show what we would have recommended
        logging.info("💡 Previously recommended jobs:")
        for job in all_promising_jobs[:5]:  # Show top 5
            score_display = f"{job['match_score_numeric']}/5" if job['match_score_numeric'] else job['match_score_text']
            logging.info("   - %s | %s: %s...", score_display, job['similarity'], job['job_text'][:70])
        
        conn.close()
        return
    
    # Sort by similarity (best matches first)
    new_promising_jobs.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Generate actionable shortlist with versioning
    SHORTLISTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = SHORTLISTS_DIR / f"smart_shortlist_{datetime.now():%Y_%m_%d_%H_%M}.md"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 🎯 Smart Job Shortlist\n\n")
            f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
            f.write(f"**Found {len(new_promising_jobs)} new promising opportunities**\n")
            if duplicate_count > 0:
                f.write(f"*({duplicate_count} duplicates from previous runs excluded)*\n\n")
            
            for i, job in enumerate(new_promising_jobs, 1):
                score_display = f"{job['match_score_numeric']}/5" if job['match_score_numeric'] else job['match_score_text']
                
                f.write(f"## {i}. {job['job_text'][:80]}...\n")
                f.write(f"- **Similarity Score**: {job['similarity']:.3f}\n")
                f.write(f"- **AI Match Score**: {score_display}\n")
                f.write(f"- **Status**: {job['application_status'].title()}\n")
                f.write(f"- **Job ID**: `{job['job_id']}`\n")
                f.write(f"- **Analyzed**: {job['processed_at'][:10]}\n")
                f.write(f"- **First Seen**: {datetime.now().strftime('%Y-%m-%d')}\n\n")
                
                # Action items based on scores
                if job['similarity'] > 0.37:
                    f.write("🚀 **High Priority** - Strong match, apply soon!\n\n")
                elif job['similarity'] > 0.34:
                    f.write("✅ **Good Match** - Worth applying\n\n")
                else:
                    f.write("🤔 **Moderate Match** - Consider if aligned with goals\n\n")
            
            # Add history section
            history = get_shortlist_history(conn, limit=5)
            if history:
                f.write("---\n\n")
                f.write("## 📜 Recently Shortlisted Jobs\n")
                f.write("*(Already processed in previous runs)*\n\n")
                for job_id, job_text, similarity, match_score, shortlisted_at in history:
                    score_display = f"{match_score}/5" if match_score else "N/A"
                    f.write(f"- `{similarity:.3f}` | {score_display} | {job_text[:60]}... (*{shortlisted_at[:10]}*)\n")
        
        logging.info("✅ Generated smart shortlist with %d NEW jobs", len(new_promising_jobs))
        logging.info("📄 Output: %s", output_file.resolve())
        
        # Show quick summary
        logging.info("🎯 New opportunities:")
        for job in new_promising_jobs:
            score_display = f"{job['match_score_numeric']}/5" if job['match_score_numeric'] else job['match_score_text']
            logging.info("   - %s | %s: %s...", score_display, job['similarity'], job['job_text'][:70])
    
    except Exception as e:
        logging.error("❌ Failed to write output file: %s", e)
    
    # Enhanced summary
    runtime = datetime.now() - start_time
    logging.info("\n📊 SUMMARY")
    logging.info("🧩 Total jobs analyzed: %d", len(data))
    logging.info("✅ Promising jobs found: %d", len(all_promising_jobs))
    logging.info("🆕 New jobs shortlisted: %d", len(new_promising_jobs))
    logging.info("♻️ Duplicates skipped: %d", duplicate_count)
    logging.info("🕒 Runtime: %s", str(runtime).split('.')[0])
    logging.info("💾 Output: %s", output_file.resolve())
    logging.info("📜 Log file: %s", log_file.resolve())
    
    conn.close()

def find_latest_json(directory):
    """Find most recent JSON file"""
    try:
        json_files = list(directory.glob("*.json"))
        return max(json_files, key=lambda x: x.stat().st_mtime) if json_files else None
    except Exception as e:
        logging.error("❌ Failed to find JSON files: %s", e)
        return None

# === UTILITY FUNCTIONS FOR MANAGING HISTORY ===
def clear_history():
    """Clear all history (useful for testing or resetting)"""
    try:
        conn = init_db()
        conn.execute("DELETE FROM sent_jobs")
        conn.commit()
        logging.info("🗑️ History cleared!")
        conn.close()
    except Exception as e:
        logging.error("❌ Failed to clear history: %s", e)

def show_history(limit=20):
    """Show recent shortlist history"""
    try:
        conn = init_db()
        history = get_shortlist_history(conn, limit)
        
        if not history:
            logging.info("No history found.")
            return
        
        logging.info("\n📜 Last %d shortlisted jobs:", len(history))
        for i, (job_id, job_text, similarity, match_score, shortlisted_at) in enumerate(history, 1):
            score_display = f"{match_score}/5" if match_score else "N/A"
            logging.info("%2d. %s | %s | %s | %s...", i, score_display, similarity, shortlisted_at[:10], job_text[:70])
        
        conn.close()
    except Exception as e:
        logging.error("❌ Failed to show history: %s", e)

if __name__ == "__main__":
    # You can add command line arguments here for history management
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "history":
            show_history()
        elif sys.argv[1] == "clear-history":
            clear_history()
        else:
            logging.info("Usage: python script.py [history|clear-history]")
    else:
        smart_job_shortlist()