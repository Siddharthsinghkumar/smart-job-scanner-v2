# Smart Job Scanner v2

Smart Job Scanner v2 is an OCR + LLM ETL pipeline that ingests newspaper PDFs and produces filtered job alerts for Telegram.

## Pipeline overview

Primary orchestration path:

1. `src/downloader/newspaper_downloader.py` (scheduled ingestion)
2. `scripts/run_pipeline.sh` (stages 01-11)

Stage flow in `scripts/run_pipeline.sh`:

1. `src/pipeline/stage01_pdf_to_images.py`
2. `src/pipeline/stage02_block_detection.py`
3. `src/pipeline/stage03_block_refiner.py`
4. `src/pipeline/stage04_ocr.py`
5. `src/pipeline/stage05_translation.py`
6. `src/pipeline/stage06_batch_builder.py`
7. `src/pipeline/stage07_llm_extraction.py --no-hybrid`
8. `src/pipeline/stage08_post_processing.py`
9. `src/pipeline/stage09_dynamic_resumes.py`
10. `src/pipeline/stage09_local_filter.py`
11. `src/pipeline/stage09_llm_filter.py`
12. `src/pipeline/stage09_shortlist.py`
13. `src/pipeline/stage10_notification.py`
14. `src/pipeline/stage11_cleanup.py`

## Setup

1. Create and activate your virtual environment.
2. Install runtime dependencies:

```bash
pip install -r requirements.txt
```

3. Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

4. Copy and configure environment variables:

```bash
cp .env.example .env
```

Required environment variables include:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `TELEGRAM_API_ID`
- `TELEGRAM_API_HASH`
- `GOOGLE_API_KEYS`

## Developer commands

- `make run` - run full pipeline
- `make dry-run` - validate execution order/config/paths without running stages
- `make test` - run pytest
- `make health` - run system diagnostics

## Validation and CI

- Local validation:

```bash
pytest -q
```

- CI workflow:
  - `.github/workflows/tests.yml`
  - triggers on `push` and `pull_request`
  - installs dependencies and runs `pytest -q`

## Runtime artifacts

- `logs/` - stage logs + structured JSONL logs (`pipeline_YYYYMMDD.jsonl`)
- `run_state/` - stage completion markers (`*.done`)
- `data/pipeline_metrics.json` - pipeline metrics and runtime counters

## Health check

Run read-only diagnostics:

```bash
python scripts/health_check.py
```

Checks include Python version, CUDA/Torch availability, Ollama availability, env variable presence, config structure, and directory write permissions.

## Release tagging

To tag a stable release manually:

```bash
git tag -a v2.0 -m "Smart Job Scanner v2 stable"
git push origin v2.0
```

## Documentation & Research

Detailed documentation and system audits are available in the repository:
- **[docs/](docs/)**: Contains high-level roadmaps and architectural blueprints.
- **[research/](research/)**: Contains in-depth technical audits, dependency matrices, system topology reviews, and LLM evaluation papers.
