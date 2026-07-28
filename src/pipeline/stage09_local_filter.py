import json
import logging
from src.utils.logging_utils import configure_logging
logger = configure_logging("stage09_local_filter")
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ───── Config ─────
BASE_DIR = Path("data/Jobs_found_final")      # newspaper job files
RESUMES_DIR = Path("data/dynamic_resumes")    # 3 resumes
OUTPUT_FILE = Path("data/shortlisted_jobs_json/shortlisted_jobs.json")
DEBUG_FILE = Path(f"data/shortlisted_jobs_json/debug_similarity_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.json")
DEFAULT_THRESHOLD = 0.3   # baseline similarity threshold

# ───── Logging Setup ─────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"local_filter_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.log"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)

# ───── Helper Functions ─────
def stable_id(text: str) -> str:
    """Generate a short stable SHA-256 digest for text (12 chars)."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return h[:12]

# --- HuggingFace direct embedding function ---
def mean_pooling(model_output, attention_mask):
    """Mean pooling — explicit HuggingFace Transformers implementation."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

def get_hf_embeddings(texts, tokenizer, model, batch_size=32):
    """Compute normalized embeddings using raw HuggingFace Transformers."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True,
                          max_length=512, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(**encoded)
        embeddings = mean_pooling(outputs, encoded['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings)
    return torch.cat(all_embeddings, dim=0)



def load_jobs():
    """Load all job postings with stable IDs and filesystem checks."""
    jobs = {}

    if not BASE_DIR.exists():
        logger.error(f"Jobs base dir not found: {BASE_DIR.resolve()}")
        return jobs

    for entry in BASE_DIR.iterdir():
        if not entry.is_dir():
            continue
        for job_file in entry.glob("*_jobs_all.txt"):
            try:
                with open(job_file, "r", encoding="utf-8") as f:
                    job_lines = [line.strip() for line in f if line.strip()]
                    for idx, line in enumerate(job_lines, 1):
                        job_text = line.lstrip("📌 ").replace("\n", " ").strip()
                        if not job_text:
                            continue
                        job_id = f"{entry.name}/{job_file.name}:{stable_id(job_text)}"
                        jobs[job_id] = job_text
            except Exception as e:
                logger.warning(f"Failed reading {job_file}: {e}")

    logger.info(f"📂 Loaded {len(jobs)} jobs from {BASE_DIR}")
    return jobs


def load_resumes():
    resumes = {}
    for resume_file in RESUMES_DIR.glob("*.txt"):
        with open(resume_file, "r", encoding="utf-8") as f:
            resumes[resume_file.name] = f.read().replace("\n", " ")
    logger.info(f"📂 Loaded {len(resumes)} resumes")
    return resumes


def parse_args():
    parser = argparse.ArgumentParser(description="Local job filter based on text similarity.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Similarity threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--debug", action="store_true", help="Enable console debug logs")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Optional batch size for encoding large job sets (default: 512)")
    return parser.parse_args()


# ───── Main Processing ─────
def main():
    args = parse_args()

    # Add console logging dynamically
    ch = logging.StreamHandler()
    ch_formatter = logging.Formatter("%(message)s")
    ch.setFormatter(ch_formatter)
    if args.debug:
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
        logger.info("🪲 Debug mode enabled — console will show all logs")
    else:
        ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    SIM_THRESHOLD = args.threshold
    logger.info(f"🔧 Using similarity threshold: {SIM_THRESHOLD}")

    logger.info("📂 Loading jobs...")
    jobs = load_jobs()
    if not jobs:
        logger.error("No jobs found, exiting.")
        return

    logger.info("📂 Loading resumes (expecting 3)...")
    resumes = load_resumes()
    if not resumes:
        logger.error("No resumes found, exiting.")
        return

    logger.info("🤖 Loading MiniLM model (SentenceTransformer)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    device = model.device
    logger.info(f"Use pytorch device_name: {device}")
    
    logger.info("🤖 Loading HF AutoModel for comparison...")
    hf_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    hf_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)

    resume_names = list(resumes.keys())
    resume_texts = list(resumes.values())
    resume_embeddings = model.encode(resume_texts, convert_to_tensor=True)
    hf_resume_embeddings = get_hf_embeddings(resume_texts, hf_tokenizer, hf_model)

    # Prepare job data for batching
    job_ids = list(jobs.keys())
    job_texts = list(jobs.values())
    batch_size = args.batch_size

    logger.info(f"📊 Encoding {len(job_texts)} jobs in batches of {batch_size}...")
    job_embeddings = []
    for i in tqdm(range(0, len(job_texts), batch_size), desc="Encoding batches (ST)"):
        batch_texts = job_texts[i:i + batch_size]
        batch_embeds = model.encode(batch_texts, convert_to_tensor=True)
        job_embeddings.append(batch_embeds)
    job_embeddings = torch.cat(job_embeddings, dim=0)
    
    logger.info("📊 Computing all similarities (vectorized matrix)...")
    all_scores = util.cos_sim(job_embeddings, resume_embeddings).cpu().numpy()
    
    logger.info("📊 Encoding and comparing with explicit HF Transformers...")
    hf_job_embeddings = get_hf_embeddings(job_texts, hf_tokenizer, hf_model, batch_size=batch_size)
    hf_all_scores = torch.mm(hf_job_embeddings, hf_resume_embeddings.transpose(0, 1)).cpu().numpy()

    shortlisted = []
    debug_similarity = {}

    for i, scores in enumerate(tqdm(all_scores, desc="Processing Matches", unit="job")):
        job_id = job_ids[i]
        job_text = job_texts[i]
        scores_dict = {name: float(scores[j]) for j, name in enumerate(resume_names)}
        best_resume = max(scores_dict, key=scores_dict.get)
        best_score = scores_dict[best_resume]
        
        hf_scores_dict = {name: float(hf_all_scores[i][j]) for j, name in enumerate(resume_names)}
        hf_best_resume = max(hf_scores_dict, key=hf_scores_dict.get)
        hf_best_score = hf_scores_dict[hf_best_resume]
        
        logger.info("score_comparison", st_best=round(best_score, 4), hf_best=round(hf_best_score, 4), delta=round(abs(best_score - hf_best_score), 4))

        debug_similarity[job_id] = {
            "similarities": {k: round(v, 3) for k, v in scores_dict.items()},
            "best": best_resume,
            "score": round(float(best_score), 3),
            "threshold": SIM_THRESHOLD,
            "len": len(job_text),
            "sample": job_text[:120] + ("..." if len(job_text) > 120 else "")
        }

        if best_score >= SIM_THRESHOLD:
            shortlisted.append({
                "job_id": job_id,
                "job_text": job_text,
                "best_resume": best_resume,
                "similarity": round(float(best_score), 3)
            })
            logger.info(f"✅ SHORTLISTED | Job: {job_id} | Best: {best_resume} ({best_score:.3f})")
        else:
            logger.debug(f"Job: {job_id} | Best: {best_resume} ({best_score:.3f}) below threshold")

    # Save debug JSON
    DEBUG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_FILE, "w", encoding="utf-8") as f:
        json.dump(debug_similarity, f, indent=2, ensure_ascii=False)
    logger.info(f"📄 Saved debug similarity info to {DEBUG_FILE}")

    # Save shortlisted jobs
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(shortlisted, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ {len(shortlisted)} shortlisted out of {len(jobs)} jobs")
    logger.info(f"📄 Saved shortlisted jobs to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
