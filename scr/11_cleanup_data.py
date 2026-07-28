#!/usr/bin/env python3
"""
11_cleanup_data.py – Clean up unnecessary intermediate files
Deletes contents of specified folders but keeps folder structure
"""

import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

# Directories and files to KEEP (preserve these completely)
KEEP_ITEMS = {
    # Database files
    "shortlist_history.db",
    "gemini_interactions.db",
    "processing_state.db",
    "processing_checkpoint.txt",
    
    # Telegram state
    "telegram",
    
    # LLM results (final analysis)
    "llm_results",
    
    # Dynamic resumes (for job matching)
    "dynamic_resumes",
    
    # Cache (important for performance)
    "cache",
    
    # GPU OCR errors (for fallback processing)
    "gpu_ocr_errors",
    
    # Shortlisted jobs JSON (final output)
    "shortlisted_jobs_json",
    
    # Jobs found final (processed results)
    "Jobs_found_final",
    
    # Shortlists directory (important)
    "shortlists"
}

# Directories to CLEAN (delete contents but keep folder structure)
CLEAN_DIRS = {
    "all_eng_text",          # Intermediate translation files
    "batch_inputs",          # Temporary batches for Ollama
    "batch_output",          # Temporary Ollama outputs  
    "job_blocks",            # Raw block detection
    "job_blocks_refined",    # Refined blocks (intermediate)
    "job_blocks_smart",      # Smart blocks (intermediate)
    "jobs_json",             # Intermediate JSON
    "output",                # Generic output (likely temp)
    "page_texts",            # Raw OCR text
    "pdf2img",               # PDF to image conversion
    "processed_pdfs",        # Already processed PDFs
    "raw_pdfs",              # Original PDFs (you can re-download)
    "refiner_skipped",       # Debug skipped blocks
    "test_data"              # Test data
}

# File patterns to delete (in any directory)
DELETE_PATTERNS = [
    #"*.tmp",
    #"*.temp",
    #"*.bak",
    #"*.cache",
    #"*.pickle",
    #"*.pkl"
]

DELETED_ITEMS = []
KEPT_ITEMS = []

def calculate_space_saved(paths):
    """Calculate total disk space used by given paths"""
    total = 0
    for p in paths:
        if p.is_file():
            total += p.stat().st_size
        elif p.is_dir():
            for f in p.rglob('*'):
                if f.is_file():
                    total += f.stat().st_size

    units = ["B","KB","MB","GB","TB"]
    idx = 0
    while total > 1024 and idx < len(units)-1:
        total /= 1024
        idx += 1

    return f"{total:.2f} {units[idx]}"

def clean_directory_contents(dir_path: Path):
    """Delete all contents of a directory but keep the directory itself"""
    try:
        if not dir_path.exists():
            return
            
        for item in dir_path.iterdir():
            if item.is_file():
                item.unlink()
                DELETED_ITEMS.append(f"📄 {item.relative_to(BASE_DIR)}")
            elif item.is_dir():
                shutil.rmtree(item)
                DELETED_ITEMS.append(f"📁 {item.relative_to(BASE_DIR)}")
                
        # Create a placeholder to ensure directory exists
        if not any(dir_path.iterdir()):
            placeholder = dir_path / ".gitkeep"
            placeholder.touch()
            KEPT_ITEMS.append(f"📁 KEPT STRUCTURE: {dir_path.relative_to(BASE_DIR)}")
            
    except Exception as e:
        print(f"⚠️ Failed to clean {dir_path}: {e}")

def safe_delete(path: Path):
    """Safely delete a file or directory"""
    try:
        if path.is_file():
            path.unlink()
            DELETED_ITEMS.append(f"📄 {path.relative_to(BASE_DIR)}")
        elif path.is_dir():
            shutil.rmtree(path)
            DELETED_ITEMS.append(f"📁 {path.relative_to(BASE_DIR)}")
    except Exception as e:
        print(f"⚠️ Failed to delete {path}: {e}")

def cleanup_directory_contents():
    """Clean up contents of specified directories"""
    print("🧹 Cleaning directory contents...")
    
    if not DATA_DIR.exists():
        print("❌ data/ folder not found!")
        return
    
    # First, handle directories we want to clean (delete contents but keep folder)
    for dir_name in CLEAN_DIRS:
        dir_path = DATA_DIR / dir_name
        if dir_path.exists():
            print(f"🗑️  Cleaning contents of: {dir_name}")
            clean_directory_contents(dir_path)
        else:
            KEPT_ITEMS.append(f"📁 KEPT STRUCTURE: {dir_path.relative_to(BASE_DIR)} (will be created if needed)")
    
    # Then, delete any loose files not in keep list
    for item in DATA_DIR.iterdir():
        if item.name in KEEP_ITEMS:
            KEPT_ITEMS.append(f"✅ KEPT: {item.relative_to(BASE_DIR)}")
            continue
            
        # Skip directories we're cleaning (already handled above)
        if item.name in CLEAN_DIRS:
            continue
            
        # Delete any other files/directories not in keep list
        if item.is_file() and item.name not in KEEP_ITEMS:
            safe_delete(item)
        elif item.is_dir() and item.name not in KEEP_ITEMS and item.name not in CLEAN_DIRS:
            safe_delete(item)

def cleanup_logs(days_to_keep=7):
    """Clean up old log files"""
    print("📋 Cleaning old log files...")
    
    if not LOG_DIR.exists():
        print("❌ logs/ folder not found!")
        return
    
    cutoff_time = datetime.now() - timedelta(days=days_to_keep)
    
    for log_file in LOG_DIR.glob("*.log"):
        try:
            log_time = datetime.fromtimestamp(log_file.stat().st_mtime)
            if log_time < cutoff_time:
                safe_delete(log_file)
            else:
                KEPT_ITEMS.append(f"✅ KEPT: {log_file.relative_to(BASE_DIR)}")
        except Exception as e:
            print(f"⚠️ Could not check {log_file}: {e}")

def cleanup_temp_files():
    """Clean up temporary files matching patterns"""
    print("🗑️ Cleaning temporary files...")
    
    for pattern in DELETE_PATTERNS:
        for temp_file in BASE_DIR.rglob(pattern):
            # Don't delete logs from the main cleanup
            if pattern == "*.log" and temp_file.parent == LOG_DIR:
                continue
            safe_delete(temp_file)

def show_summary():
    """Show cleanup summary"""
    print("\n" + "="*50)
    print("🧹 CLEANUP SUMMARY")
    print("="*50)
    
    if DELETED_ITEMS:
        print(f"✅ Removed {len(DELETED_ITEMS)} items:")
        for item in DELETED_ITEMS[:20]:  # Show first 20 items
            print(f"   {item}")
        if len(DELETED_ITEMS) > 20:
            print(f"   ... and {len(DELETED_ITEMS) - 20} more items")
    else:
        print("ℹ️ No items were deleted.")
    
    if KEPT_ITEMS:
        print(f"\n📦 Kept {len(KEPT_ITEMS)} essential items and structures:")
        for item in KEPT_ITEMS[:15]:  # Show first 15 kept items
            print(f"   {item}")
        if len(KEPT_ITEMS) > 15:
            print(f"   ... and {len(KEPT_ITEMS) - 15} more items")

def main():
    """Main cleanup workflow"""
    print("🚀 Starting data cleanup utility...")
    print(f"📁 Base directory: {BASE_DIR}")
    
    # Collect paths that will be deleted for space calculation
    to_delete = []
    for d in CLEAN_DIRS:
        dir_path = DATA_DIR / d
        if dir_path.exists():
            to_delete.append(dir_path)
    
    # Calculate and store space before cleanup
    space_before = "0 B"
    if to_delete:
        space_before = calculate_space_saved(to_delete)
        print("💾 Space before cleanup:", space_before)
    
    # Show what will be kept and cleaned
    print("\n📦 WILL KEEP (completely):")
    for item in sorted(KEEP_ITEMS):
        print(f"   ✅ {item}")
    
    print("\n🗑️ WILL CLEAN (delete contents, keep folder):")
    for item in sorted(CLEAN_DIRS):
        print(f"   🧹 {item}")
    
    # Confirm before proceeding
    response = input("\n❓ Proceed with cleanup? This cannot be undone! (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Cleanup cancelled.")
        return
    
    # Perform cleanup operations
    cleanup_directory_contents()
    cleanup_logs(days_to_keep=7)
    cleanup_temp_files()
    
    # Show results
    show_summary()
    
    # Display the actual space that was freed (stored before deletion)
    if to_delete:
        print(f"💾 Space freed: {space_before}")
    
    print("\n🎉 Cleanup completed!")
    print("📁 Folder structure preserved for future processing")

if __name__ == "__main__":
    main()