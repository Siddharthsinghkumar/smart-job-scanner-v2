.PHONY: run dry-run test-run test health
PYTHON ?= ./4_env/bin/python

run:
	$(PYTHON) scripts/run_pipeline.py

dry-run:
	$(PYTHON) scripts/run_pipeline.py --dry-run

test-run:
	$(PYTHON) scripts/run_pipeline.py --test-run

test:
	$(PYTHON) -m pytest -q

health:
	$(PYTHON) scripts/health_check.py
