.PHONY: run dry-run test health

run:
	./scripts/run_pipeline.sh

dry-run:
	./scripts/run_pipeline.sh --dry-run

test:
	./4_env/bin/python -m pytest -q

health:
	./4_env/bin/python scripts/health_check.py
