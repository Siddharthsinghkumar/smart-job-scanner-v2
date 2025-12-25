#!/usr/bin/env python3
"""
simple_gemini.py - Fixed Gemini client with working models
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
import google.generativeai as genai

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_gemini")

class SimpleGeminiClient:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        # Use the working models from your test
        self.model_names = [
            #'gemini-2.5-flash',  # Fast and efficient
            'gemini-2.5-flash-lite',  # Even faster
            #'gemini-2.5-pro',  # More capable but slower
        ]
        self.current_model_index = 0
        
    def generate_content(self, prompt, max_retries=3):
        last_error = None
        
        # Try each model in order
        for model_index in range(len(self.model_names)):
            model_name = self.model_names[model_index]
            logger.info(f"Trying model: {model_name}")
            
            for attempt in range(max_retries):
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    logger.info(f"✅ Success with model: {model_name}")
                    return response.text
                except Exception as e:
                    last_error = e
                    if "quota" in str(e).lower() or "rate" in str(e).lower():
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1})")
                        time.sleep(wait_time)
                        continue
                    else:
                        # If it's not a rate limit error, try next model
                        logger.warning(f"Model {model_name} failed: {e}")
                        break
        
        raise Exception(f"All models and retries exhausted. Last error: {last_error}")

def load_config():
    """Load config from gemini_config.json"""
    config_file = Path("gemini_config.json")
    if not config_file.exists():
        raise FileNotFoundError("gemini_config.json not found")
    
    with open(config_file, 'r') as f:
        return json.load(f)

def analyze_job_with_gemini(job_data, resume_text, api_key):
    """Analyze a single job with Gemini"""
    client = SimpleGeminiClient(api_key)
    
    prompt = f"""
Analyze this job posting and compare it with the candidate's resume:

JOB POSTING:
"{job_data['job_text']}"

CANDIDATE'S RESUME EXCERPT:
{resume_text[:3000]}

Please provide a structured analysis with:

1. **Job Identification**: 
   - Probable job title and company
   - Industry/sector

2. **Skills Match Analysis**:
   - Key required skills from job posting
   - How candidate's skills align (match/mismatch)
   - Any notable gaps

3. **Relevance Assessment**:
   - Overall relevance score (1-5 scale)
   - Brief explanation of the score

4. **Recommendation**:
   - CLEARLY state: "RECOMMENDED" or "NOT RECOMMENDED"
   - 2-3 key reasons for the recommendation

5. **Next Steps** (if recommended):
   - Suggested focus areas for application
   - Any additional preparation needed

Be concise, practical, and focus on actionable insights. Avoid fluff.
"""
    
    try:
        response = client.generate_content(prompt)
        return {
            "job_id": job_data["job_id"],
            "job_text": job_data["job_text"],
            "resume_used": job_data["best_resume"],
            "similarity": job_data["similarity"],
            "gemini_response": response,
            "status": "success"
        }
    except Exception as e:
        return {
            "job_id": job_data["job_id"],
            "job_text": job_data["job_text"],
            "resume_used": job_data["best_resume"],
            "similarity": job_data["similarity"],
            "gemini_response": f"ERROR: {str(e)}",
            "status": "error"
        }