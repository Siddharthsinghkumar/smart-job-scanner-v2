import os
import re
import difflib
import logging
from src.utils.logging_utils import configure_logging
logger = configure_logging("stage08_post_processing")
import signal
import sys
from pathlib import Path
from datetime import datetime

# ───── Config ─────
BASE_DIR = Path("data/Jobs_found_final")

# ───── Logging Setup ─────
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
log_name = f"deduplication_{timestamp}.log"

# File logger (all details)
s [%(levelname)s] %(message)s",
    level=logging.INFO,
)

# Console logger (only clean summaries)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(console)


# ───── Deduplication Helper ─────
def deduplicate_jobs(jobs, threshold=0.85):
    """Deduplicate jobs using string similarity. 
    Logs duplicates to file only, not terminal."""
    unique_jobs = []
    logger = logging.getLogger()
    file_handler = logger.handlers[0]  # first handler = file logger

    for job in jobs:
        duplicate_found = False
        for kept in unique_jobs:
            ratio = difflib.SequenceMatcher(None, job, kept).ratio()
            if ratio >= threshold:
                # Log duplicates only in file
                file_handler.emit(
                    logging.LogRecord(
                        name="dedup",
                        level=logging.INFO,
                        pathname=__file__,
                        lineno=0,
                        msg=f"[DUPLICATE] Removed: \"{job}\" ≈ Kept: \"{kept}\" (similarity {ratio:.2f})",
                        args=None,
                        exc_info=None,
                    )
                )
                duplicate_found = True
                break
        if not duplicate_found:
            unique_jobs.append(job)
    return unique_jobs


# ───── Main ─────
def main():
    total_raw_all = 0
    total_unique_all = 0

    for newspaper_dir in BASE_DIR.iterdir():
        if newspaper_dir.is_dir():
            issue_prefixes = set(f.name.split("_p")[0] for f in newspaper_dir.glob("*_jobs.txt"))
            for issue_prefix in issue_prefixes:
                issue_files = list(newspaper_dir.glob(f"{issue_prefix}*_jobs.txt"))
                all_jobs = []

                for file in issue_files:
                    try:
                        with open(file, "r", encoding="utf-8") as f:
                            all_jobs.extend(line.strip() for line in f if line.strip())
                    except Exception as e:
                        logging.error(f"❌ Failed to read {file}: {e}")

                if not all_jobs:
                    continue

                deduped_jobs = deduplicate_jobs(all_jobs)

                out_file = newspaper_dir / f"{issue_prefix}_jobs_all.txt"
                with open(out_file, "w", encoding="utf-8") as f:
                    for idx, job in enumerate(deduped_jobs, 1):
                        clean_job = job.lstrip("📌 ").strip()
                        f.write(f"{idx}. {clean_job}\n")

                total_raw = len(all_jobs)
                total_unique = len(deduped_jobs)
                reduction = 100 * (1 - total_unique / total_raw)

                msg = (
                    f"{newspaper_dir.name}/{issue_prefix}: "
                    f"{total_raw} raw → {total_unique} unique "
                    f"({reduction:.1f}% duplicates removed)"
                )

                logging.info(msg)

                total_raw_all += total_raw
                total_unique_all += total_unique

    if total_raw_all > 0:
        reduction_all = 100 * (1 - total_unique_all / total_raw_all)
        summary_msg = (
            f"\nTOTAL: {total_raw_all} raw → {total_unique_all} unique "
            f"({reduction_all:.1f}% duplicates removed overall)"
        )
        # ✅ Log once (console + file), no print() → avoids double reporting
        logging.info(summary_msg)


# ───── Entrypoint ─────
def handle_exit(sig, frame):
    print("\n⚠️ Exit signal received. Exiting...")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    main()
