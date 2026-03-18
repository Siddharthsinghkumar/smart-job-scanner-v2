import os
import logging
from pathlib import Path
import requests
from datetime import datetime

# ───── Logging Setup ─────
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
log_name = f"batch_run_{timestamp}.log"

logging.basicConfig(
    filename=log_dir / log_name,
    filemode='a',
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# ───── Settings ─────
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

batch_input_dir = Path("data/batch_inputs")
output_dir = Path("data/batch_output")
output_dir.mkdir(parents=True, exist_ok=True)

KEYWORDS = [
    "job", "recruitment", "intern", "hiring", "walk-in",
    "vacancy", "apply", "eligibility", "bio-data", "post"
]

# ───── Prompt Template ─────
PROMPT_TEMPLATE = """
[INSTRUCTION]
You are an assistant that extracts job opportunities from newspapers.

Your task is to:
- Identify all job postings, recruitment announcements, internship offers, and walk-in interview notices
- Ignore tenders, finance news, astrology, sports, or unrelated content
- Return only the job opportunities in the format shown below

The job ads may be short or unstructured. Still extract them if they mention positions, hiring, application, eligibility, walk-in dates, or how to apply.

Return in this format:
- 📌 *[Title or Employer]*: [Position, salary, deadline, contact method]

Examples:
- 📌 *Hiring of Staff for URC 35 Int Bde*: Applications invited for 2 x Multi-Tasking Staff (MTS), salary Rs. 22,500/month. Apply by 23 Jul 2025.
- 📌 *Internship at IIT Patna*: Final year students eligible. Last date: 15 Aug 2025.
- 📌 *Private Firm in Noida*: Walk-in interview for Account Executive, B.Com required, salary ₹28,000/month. Interview on 1st Aug 2025.
- 📌 *ONGC Ltd*: Apprenticeship program for ITI and Diploma holders. Online registration open. Last date: 5 Aug 2025.
- 📌 *Haryana Public Service Commission*: Recruitment for 20 Assistant Engineers, apply online by 10 Aug 2025.

[NEWSPAPER_CONTENT]
{content}
[/NEWSPAPER_CONTENT]
""".strip()

# ───── Model Query Function ─────
def query_model(prompt_text):
    try:
        response = requests.post(
            OLLAMA_ENDPOINT,
            json={"model": MODEL_NAME, "prompt": prompt_text, "stream": False},
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.RequestException as e:
        logging.error(f"❌ Model query failed: {e}")
        return ""

# ───── Main Loop ─────
def main():
    logging.info("🚀 Starting Ollama batch processing (API mode)...")

    batch_files = sorted(batch_input_dir.glob("*.txt"))
    if not batch_files:
        logging.warning("⚠️ No batch text files found in data/batch_inputs/")
        return

    for batch_file in batch_files:
        logging.info(f"📄 Processing: {batch_file.name}")
        
        try:
            raw_content = batch_file.read_text(encoding="utf-8")
            prompt = PROMPT_TEMPLATE.format(content=raw_content)

            output_text = query_model(prompt).strip()

            if not output_text:
                logging.warning(f"⚠️ Empty model response for {batch_file.name}")
                continue

            # Save full response
            response_path = output_dir / (batch_file.stem + "_response.txt")
            response_path.write_text(output_text, encoding="utf-8")
            logging.info(f"[💾] Saved response: {response_path.name}")

            # Filter for job lines
            lines = output_text.splitlines()
            relevant = [line for line in lines if any(k in line.lower() for k in KEYWORDS)]

            if relevant:
                jobs_path = output_dir / (batch_file.stem + "_jobs.txt")
                jobs_path.write_text("\n".join(relevant), encoding="utf-8")
                logging.info(f"[🎯] Filtered jobs: {jobs_path.name}")
            else:
                logging.warning(f"[🕵️] No job info detected in: {batch_file.name}")

        except Exception as e:
            logging.error(f"❌ Error processing {batch_file.name}: {e}")

    logging.info("✅ Finished all batch files.")

# ───── Entrypoint ─────
if __name__ == "__main__":
    main()
