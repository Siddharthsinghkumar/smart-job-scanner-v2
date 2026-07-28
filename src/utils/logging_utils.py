"""Centralized structured logging configuration for all pipeline stages.
Repurposed from dead scaffolding (F-033) — now the single source of truth."""
import structlog
import logging
import sys
from pathlib import Path

def configure_logging(stage_name: str, log_dir: Path | None = None, debug: bool = False):
    """Configure structlog with JSON output for a given stage.
    Call this once at the top of each stage's main()."""
    log_dir = log_dir or Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure structlog processors
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.DEBUG if debug else logging.INFO
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(
            file=open(log_dir / f"{stage_name}.jsonl", "a", encoding="utf-8")
        ),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger(stage=stage_name)
