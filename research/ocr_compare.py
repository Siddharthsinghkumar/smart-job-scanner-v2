import os
import re
import collections
import langdetect
import pandas as pd

CPU_DIR = "/home/sidd/project/smart-job-scanner-v2/data/test_data/cpu"
GPU_DIR = "/home/sidd/project/smart-job-scanner-v2/data/test_data/gpu"

def read_texts(base_dir):
    texts = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".txt"):
                path = os.path.join(root, f)
                rel = os.path.relpath(path, base_dir)  # keep subfolder structure
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    texts[rel] = fh.read()
    return texts

def text_stats(text):
    words = re.findall(r"\w+", text, re.UNICODE)
    chars = len(text)
    word_count = len(words)
    
    # Junk: single char words or nonsense tokens
    junk_tokens = [w for w in words if len(w) == 1]
    junk_ratio = len(junk_tokens) / word_count if word_count else 0
    
    # Avg word length
    avg_word_len = sum(len(w) for w in words) / word_count if word_count else 0
    
    # Language detection (on a sample of text to save time)
    try:
        lang = langdetect.detect(text[:2000]) if text.strip() else "unknown"
    except:
        lang = "unknown"
    
    return {
        "chars": chars,
        "words": word_count,
        "junk_ratio": round(junk_ratio, 3),
        "avg_word_len": round(avg_word_len, 2),
        "lang_guess": lang
    }

def compare(cpu_texts, gpu_texts):
    rows = []
    all_keys = set(cpu_texts.keys()) | set(gpu_texts.keys())
    for key in sorted(all_keys):
        cpu_stats = text_stats(cpu_texts.get(key, ""))
        gpu_stats = text_stats(gpu_texts.get(key, ""))
        
        row = {
            "file": key,
            "cpu_words": cpu_stats["words"],
            "gpu_words": gpu_stats["words"],
            "cpu_junk": cpu_stats["junk_ratio"],
            "gpu_junk": gpu_stats["junk_ratio"],
            "cpu_avg_len": cpu_stats["avg_word_len"],
            "gpu_avg_len": gpu_stats["avg_word_len"],
            "cpu_lang": cpu_stats["lang_guess"],
            "gpu_lang": gpu_stats["lang_guess"],
        }
        rows.append(row)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print("Reading CPU OCR texts...")
    cpu_texts = read_texts(CPU_DIR)
    print("Reading GPU OCR texts...")
    gpu_texts = read_texts(GPU_DIR)
    
    df = compare(cpu_texts, gpu_texts)
    
    # Save to CSV for detailed review
    out_file = "/home/sidd/project/smart-job-scanner-v2/ocr_comparison_report.csv"
    df.to_csv(out_file, index=False)
    
    print(f"✅ Comparison report saved: {out_file}")
    print(df.head(10))
