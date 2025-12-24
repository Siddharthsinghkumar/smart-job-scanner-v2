#!/usr/bin/env python3
# ==============================================
# Script: 9-1_dynamic_resumes_full.py - SIMPLIFIED
# Purpose: Dynamic resume builder with working GitHub token
# ==============================================

import os
import sys
import signal
import logging
import time
import json
from pathlib import Path
from datetime import datetime
import pdfplumber
import re
from github import Github, Auth

# ───── CONFIG ─────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = BASE_DIR / "resumes"
OUTPUT_DIR = DATA_DIR / "dynamic_resumes"
LOG_DIR = BASE_DIR / "logs"
CACHE_DIR = DATA_DIR / "cache"

#Use your own
EE_GITHUB_USER = "*******" #Use your GitHub profile
AIML_GITHUB_USER = "#########"  #Use your GitHub profile
EE_PDF = PDF_DIR / "EE_resume.pdf"
AIML_PDF = PDF_DIR / "AI_ML_resume.pdf"

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ───── Logging Setup ─────
log_file = LOG_DIR / f"dynamic_resumes_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler()
    ]
)
logging.info(f"📂 Log file: {log_file}")

# ───── Simple GitHub Auth Setup ─────
def setup_github_client():
    """Simple GitHub client setup without complex rate limit checks"""
    token = os.getenv("AIML_GITHUB_TOKEN")
    
    if not token:
        logging.warning("❌ No AIML_GITHUB_TOKEN found! Using public access (60 req/hr limit)")
        return Github(), False
    
    try:
        g = Github(auth=Auth.Token(token))
        # Simple test - get current user
        user = g.get_user()
        logging.info(f"🔑 GitHub token authenticated as: {user.login}")
        return g, True
    except Exception as e:
        logging.error(f"❌ GitHub token authentication failed: {e}")
        logging.warning("🔄 Falling back to public access")
        return Github(), False

# Initialize GitHub client
g, using_token = setup_github_client()

# ───── GitHub Functions ─────
def extract_github_skills_simple(username):
    """Simple GitHub skills extraction with basic error handling"""
    cache_file = CACHE_DIR / f"github_skills_{username}.json"
    all_skills = ""
    
    # Check cache first
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            logging.info(f"💾 Using cached GitHub data for {username}")
            return cached_data.get("skills_text", "")
        except Exception as e:
            logging.warning(f"⚠️ Cache load failed: {e}")
    
    try:
        user = g.get_user(username)
        repos = list(user.get_repos())
        
        logging.info(f"🔹 Found {len(repos)} public repos for {username}")
        
        for i, repo in enumerate(repos):
            try:
                # Small delay to be API-friendly
                if i > 0:
                    time.sleep(0.5)
                
                repo_text = f"Project: {repo.name} | Language: {repo.language or 'N/A'} | Description: {repo.description or ''}"
                
                # Try to get topics (only with token)
                if using_token:
                    try:
                        topics = repo.get_topics()
                        if topics:
                            repo_text += f" | Topics: {', '.join(topics)}"
                    except Exception:
                        pass  # Silently skip if topics fail
                
                # Try to get README excerpt
                try:
                    readme = repo.get_readme()
                    readme_content = readme.decoded_content.decode()
                    # Take first 200 chars of README
                    excerpt = readme_content[:200].replace('\n', ' ').strip()
                    if excerpt:
                        repo_text += f" | README: {excerpt}..."
                except Exception as e:
                    # 404 means no README - that's normal
                    if "404" not in str(e):
                        logging.debug(f"⚠️ Could not get README for {repo.name}: {e}")
                
                all_skills += repo_text + "\n\n"
                
            except Exception as e:
                logging.warning(f"⚠️ Failed to process repo {repo.name}: {e}")
                continue
        
        # Cache the results
        try:
            cache_data = {
                "username": username,
                "last_updated": datetime.now().isoformat(),
                "repo_count": len(repos),
                "skills_text": all_skills
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            logging.info(f"💾 Cached GitHub data for {username}")
        except Exception as e:
            logging.warning(f"⚠️ Failed to cache data: {e}")
        
        logging.info(f"🧠 Generated GitHub skills for {username}: {len(all_skills.splitlines())} lines")
        return all_skills
        
    except Exception as e:
        logging.error(f"❌ GitHub extraction failed for {username}: {e}")
        return ""

# ───── PDF Processing ─────
def pdf_to_text(pdf_path):
    """Convert PDF to plain text"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text.strip())
        logging.info(f"✅ PDF converted: {pdf_path.name}")
        return text
    except Exception as e:
        logging.error(f"❌ Failed to read PDF {pdf_path}: {e}")
        return ""

def save_text(file_path, text):
    """Save text to file"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        logging.info(f"✅ Saved: {file_path.name}")
    except Exception as e:
        logging.error(f"❌ Failed to save {file_path}: {e}")

def main():
    logging.info("🚀 Starting dynamic resume build...")
    logging.info(f"🔑 GitHub authentication: {'TOKEN' if using_token else 'PUBLIC'}")

    # Check if PDFs exist
    if not EE_PDF.exists():
        logging.error(f"❌ PDF not found: {EE_PDF}")
        return
    if not AIML_PDF.exists():
        logging.error(f"❌ PDF not found: {AIML_PDF}")
        return

    # 1️⃣ Convert PDFs
    logging.info("📄 Processing PDF resumes...")
    ee_resume_text = pdf_to_text(EE_PDF)
    aiml_resume_text = pdf_to_text(AIML_PDF)

    if not ee_resume_text or not aiml_resume_text:
        logging.error("❌ PDF processing failed")
        return

    # 2️⃣ Extract GitHub skills
    logging.info("🔍 Fetching GitHub skills...")
    ee_github_skills = extract_github_skills_simple(EE_GITHUB_USER)
    aiml_github_skills = extract_github_skills_simple(AIML_GITHUB_USER)

    # 3️⃣ Build combined resumes
    logging.info("🔨 Building combined resumes...")
    ee_combined = (ee_resume_text + "\n\n" + ee_github_skills).strip()
    aiml_combined = (aiml_resume_text + "\n\n" + aiml_github_skills).strip()
    hybrid_combined = (
        ee_resume_text + "\n\n" +
        aiml_resume_text + "\n\n" +
        ee_github_skills + aiml_github_skills
    ).strip()

    # 4️⃣ Save to OUTPUT_DIR
    logging.info("💾 Saving resume files...")
    save_text(OUTPUT_DIR / "dynamic_resume_ee.txt", ee_combined)
    save_text(OUTPUT_DIR / "dynamic_resume_aiml.txt", aiml_combined)
    save_text(OUTPUT_DIR / "dynamic_resume_hybrid.txt", hybrid_combined)

    # 5️⃣ Summary
    logging.info("\n📊 FINAL SUMMARY:")
    logging.info(f" - EE-focused resume: {len(ee_combined)} chars")
    logging.info(f" - AI/ML-focused resume: {len(aiml_combined)} chars") 
    logging.info(f" - Hybrid resume: {len(hybrid_combined)} chars")
    logging.info(f"✅ Dynamic resume build completed!")
    logging.info(f"📁 Output directory: {OUTPUT_DIR.resolve()}")

# ───── Entrypoint ─────
def handle_exit(sig, frame):
    logging.info("⚠️ Exit signal received. Exiting gracefully...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    main()