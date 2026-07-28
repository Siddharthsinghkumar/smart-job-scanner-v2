#!/usr/bin/env python3
"""
9-4_llm_search_rest.py - Working version with Gemini 2.5 models
"""

import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from simple_gemini import load_config, analyze_job_with_gemini

# Paths
SHORTLIST_PATH = Path("data/shortlisted_jobs_json/shortlisted_jobs.json")
RESUME_DIR = Path("data/dynamic_resumes")
RESULT_DIR = Path("data/llm_results")
RESULT_JSON = RESULT_DIR / f"llm_job_analysis_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.json"

# Create output directory
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("📄 Loading shortlisted jobs...")
    with open(SHORTLIST_PATH, "r", encoding="utf-8") as f:
        shortlisted = json.load(f)
    
    print(f"🔑 Loading Gemini config...")
    config = load_config()
    api_keys = config.get("google_api_keys", [])
    
    if not api_keys:
        raise Exception("No API keys found in config")
    
    # Use the first API key
    api_key = api_keys[0]
    print(f"✅ Using API key: {api_key[:20]}...")
    print(f"🎯 Using Gemini 2.5 models (flash, flash-lite, pro)")
    
    results = []
    successful_count = 0
    
    for i, job_data in enumerate(tqdm(shortlisted, desc="Analyzing jobs")):
        resume_path = RESUME_DIR / job_data["best_resume"]
        
        if not resume_path.exists():
            print(f"⚠️ Resume not found: {resume_path}")
            continue
            
        resume_text = resume_path.read_text(encoding="utf-8")
        
        print(f"\n🔍 Analyzing job {i+1}/{len(shortlisted)}...")
        try:
            result = analyze_job_with_gemini(job_data, resume_text, api_key)
            results.append(result)
            
            if result["status"] == "success":
                successful_count += 1
                print(f"✅ Job {i+1} analyzed successfully")
                # Extract first line of response for preview
                first_line = result["gemini_response"].split('\n')[0][:100]
                print(f"   Preview: {first_line}...")
            else:
                print(f"❌ Job {i+1} failed: {result['gemini_response']}")
                
        except Exception as e:
            print(f"💥 Job {i+1} crashed: {e}")
            results.append({
                "job_id": job_data["job_id"],
                "job_text": job_data["job_text"],
                "resume_used": job_data["best_resume"],
                "similarity": job_data["similarity"],
                "gemini_response": f"CRASH: {str(e)}",
                "status": "error"
            })
        
        # Rate limiting - be more generous with the new models
        if i < len(shortlisted) - 1:  # Don't sleep after the last job
            print("⏳ Waiting 3 seconds before next request...")
            time.sleep(3)
    
    # Save results
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Analysis completed!")
    print(f"📊 Results: {successful_count} successful, {len(results) - successful_count} failed")
    print(f"💾 Saved to: {RESULT_JSON}")
    
    # Show sample of successful responses
    if successful_count > 0:
        print(f"\n📋 Sample of successful analyses:")
        for i, result in enumerate(results[:2]):  # Show first 2
            if result["status"] == "success":
                print(f"\n--- Job {i+1} ---")
                print(result["gemini_response"][:500] + "...")

if __name__ == "__main__":
    main()