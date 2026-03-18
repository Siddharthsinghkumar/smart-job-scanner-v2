#!/usr/bin/env python3
"""
job_analysis_query.py - Advanced Querying for Job Analysis Database
Purpose: Query, analyze, and report on job analysis results
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/llm_results/llm_job_analysis.db")

def get_db_connection():
    """Get database connection with row factory"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn

def show_database_stats():
    """Show comprehensive database statistics"""
    if not DB_PATH.exists():
        print("❌ Database file not found. Run the analysis script first.")
        return
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Basic counts
    cur.execute("SELECT COUNT(*) as total FROM job_analysis")
    total = cur.fetchone()["total"]
    
    cur.execute("SELECT COUNT(*) as success FROM job_analysis WHERE status='success'")
    successful = cur.fetchone()["success"]
    
    cur.execute("SELECT COUNT(*) as recommended FROM job_analysis WHERE gemini_response LIKE '%RECOMMENDED%'")
    recommended = cur.fetchone()["recommended"]
    
    print("📊 DATABASE STATISTICS")
    print("=" * 50)
    print(f"Total analyses: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Recommended jobs: {recommended}")
    print(f"Success rate: {successful/total*100:.1f}%")
    print(f"Recommendation rate: {recommended/total*100:.1f}%")
    
    # Model usage
    print(f"\n🤖 MODEL USAGE:")
    cur.execute("SELECT model_used, COUNT(*) as count FROM job_analysis GROUP BY model_used ORDER BY count DESC")
    for row in cur.fetchall():
        print(f"   {row['model_used']}: {row['count']} jobs")
    
    conn.close()

def show_recommended_jobs(threshold=0.3, limit=10):
    """Show highly recommended jobs above similarity threshold"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT job_id, resume_used, similarity_score, gemini_response,
               analysis_timestamp, model_used
        FROM job_analysis 
        WHERE similarity_score > ? 
        AND gemini_response LIKE '%RECOMMENDED%'
        AND status = 'success'
        ORDER BY similarity_score DESC
        LIMIT ?
    """, (threshold, limit))
    
    print(f"\n🎯 HIGHLY RECOMMENDED JOBS (Top {limit})")
    print("=" * 60)
    
    jobs = cur.fetchall()
    if not jobs:
        print("No recommended jobs found above threshold.")
        conn.close()
        return
    
    for i, job in enumerate(jobs, 1):
        print(f"\n{i}. 📊 Score: {job['similarity_score']:.3f} | 📝 {job['resume_used']}")
        print(f"   🆔 {job['job_id']}")
        print(f"   🤖 Model: {job['model_used']}")
        print(f"   🕒 {job['analysis_timestamp'][:16]}")
        
        # Extract key insights from response
        response = job['gemini_response']
        lines = response.split('\n')
        key_points = [line for line in lines if any(word in line.upper() for word in ['RECOMMEND', 'SCORE', 'STRENGTH', 'MATCH'])]
        
        if key_points:
            print("   💡 Key Insights:")
            for point in key_points[:3]:  # Show top 3 insights
                print(f"      • {point.strip()}")
    
    conn.close()

def resume_performance_analysis():
    """Analyze which resume performs best"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT resume_used,
               COUNT(*) as total_jobs,
               AVG(similarity_score) as avg_score,
               SUM(CASE WHEN gemini_response LIKE '%RECOMMENDED%' THEN 1 ELSE 0 END) as recommended_count,
               MAX(similarity_score) as best_score
        FROM job_analysis
        WHERE status = 'success'
        GROUP BY resume_used
        ORDER BY avg_score DESC
    """)
    
    print(f"\n📈 RESUME PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    for row in cur.fetchall():
        resume = row['resume_used']
        total = row['total_jobs']
        avg_score = row['avg_score']
        recommended = row['recommended_count']
        best_score = row['best_score']
        
        if total > 0:
            success_rate = (recommended / total) * 100
            print(f"\n📄 {resume}:")
            print(f"   📊 Total Jobs: {total}")
            print(f"   ⭐ Average Score: {avg_score:.3f}")
            print(f"   🏆 Best Score: {best_score:.3f}")
            print(f"   ✅ Recommended: {recommended}/{total} ({success_rate:.1f}%)")
    
    conn.close()

def newspaper_analysis():
    """Analyze which newspapers have the best opportunities"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT 
            CASE 
                WHEN INSTR(job_id, '/') > 0 THEN SUBSTR(job_id, 1, INSTR(job_id, '/')-1)
                ELSE 'Unknown'
            END as newspaper,
            COUNT(*) as job_count,
            AVG(similarity_score) as avg_similarity,
            SUM(CASE WHEN gemini_response LIKE '%RECOMMENDED%' THEN 1 ELSE 0 END) as recommended_count
        FROM job_analysis 
        WHERE status = 'success'
        GROUP BY newspaper
        ORDER BY avg_similarity DESC
    """)
    
    print(f"\n📰 NEWSPAPER/SOURCE ANALYSIS")
    print("=" * 50)
    
    for row in cur.fetchall():
        newspaper = row['newspaper']
        count = row['job_count']
        avg_similarity = row['avg_similarity']
        recommended = row['recommended_count']
        
        if count > 0:
            rec_rate = (recommended / count) * 100
            print(f"   {newspaper}:")
            print(f"      📊 Jobs: {count} | ⭐ Avg Score: {avg_similarity:.3f}")
            print(f"      ✅ Recommended: {recommended}/{count} ({rec_rate:.1f}%)")
    
    conn.close()

def export_to_json(output_path=None):
    """Export database to JSON for sharing/backup"""
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"job_analysis_export_{timestamp}.json"
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT * FROM job_analysis")
    results = []
    
    for row in cur.fetchall():
        result_dict = {}
        for key in row.keys():
            result_dict[key] = row[key]
        results.append(result_dict)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported {len(results)} records to: {output_path}")
    conn.close()

def get_detailed_analysis(job_id=None):
    """Get detailed analysis for a specific job or the most recent"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    if job_id:
        cur.execute("SELECT * FROM job_analysis WHERE job_id = ?", (job_id,))
    else:
        cur.execute("SELECT * FROM job_analysis ORDER BY id DESC LIMIT 1")
    
    row = cur.fetchone()
    if row:
        print(f"\n🔍 DETAILED ANALYSIS:")
        print("=" * 60)
        print(f"Job ID: {row['job_id']}")
        print(f"Resume Used: {row['resume_used']}")
        print(f"Similarity Score: {row['similarity_score']:.3f}")
        print(f"Model: {row['model_used']}")
        print(f"Status: {row['status']}")
        print(f"Timestamp: {row['analysis_timestamp']}")
        print(f"Application Status: {row['application_status']}")
        print(f"\n📄 FULL ANALYSIS:")
        print("-" * 40)
        print(row['gemini_response'])
    else:
        print("❌ No analysis found")
    
    conn.close()

def update_application_status(job_id, status):
    """Update application status for a job"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    valid_statuses = ['pending', 'applied', 'interview', 'rejected', 'accepted', 'withdrawn']
    if status not in valid_statuses:
        print(f"❌ Invalid status. Use one of: {', '.join(valid_statuses)}")
        return
    
    cur.execute("UPDATE job_analysis SET application_status = ? WHERE job_id = ?", (status, job_id))
    
    if cur.rowcount > 0:
        conn.commit()
        print(f"✅ Updated job {job_id} to status: {status}")
    else:
        print(f"❌ Job not found: {job_id}")
    
    conn.close()

def main():
    """Main menu for the query helper"""
    print("🔍 Smart Job Scanner - Database Query Helper")
    print("=" * 50)
    
    if not DB_PATH.exists():
        print("❌ Database not found. Please run the analysis script first.")
        return
    
    while True:
        print(f"\n📊 QUERY OPTIONS:")
        print("1. Database Statistics")
        print("2. Show Recommended Jobs")
        print("3. Resume Performance")
        print("4. Newspaper Analysis") 
        print("5. Export to JSON")
        print("6. Get Detailed Analysis")
        print("7. Update Application Status")
        print("8. Run All Reports")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-8): ").strip()
        
        if choice == '1':
            show_database_stats()
        elif choice == '2':
            threshold = float(input("Enter similarity threshold (0.1-1.0) [0.3]: ") or "0.3")
            show_recommended_jobs(threshold)
        elif choice == '3':
            resume_performance_analysis()
        elif choice == '4':
            newspaper_analysis()
        elif choice == '5':
            export_to_json()
        elif choice == '6':
            job_id = input("Enter job ID (or leave blank for most recent): ").strip()
            get_detailed_analysis(job_id if job_id else None)
        elif choice == '7':
            job_id = input("Enter job ID: ").strip()
            status = input("Enter new status (pending/applied/interview/rejected/accepted/withdrawn): ").strip()
            update_application_status(job_id, status)
        elif choice == '8':
            show_database_stats()
            show_recommended_jobs()
            resume_performance_analysis()
            newspaper_analysis()
        elif choice == '0':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()