#!/usr/bin/env python3
"""Centralized path config loader for pipeline stages."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline_paths.json"

DEFAULT_PATHS = {
    "pdf_input": "data/raw_pdfs",
    "images_output": "data/pdf2img",
    "blocks_output": "data/job_blocks_smart",
    "refined_output": "data/job_blocks_refined",
    "detections_output": "run_state/detections",
}


def _merge_paths(config_paths: dict[str, Any] | None) -> dict[str, str]:
    merged = dict(DEFAULT_PATHS)
    if isinstance(config_paths, dict):
        for key, value in config_paths.items():
            if value is not None:
                merged[str(key)] = str(value)
    return merged


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(config_path) if config_path else Path(
        os.getenv("PIPELINE_PATHS_CONFIG", str(DEFAULT_CONFIG_PATH))
    )

    payload: dict[str, Any] = {}
    if cfg_path.exists():
        try:
            loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            payload = {}

    payload_paths = payload.get("paths") if isinstance(payload, dict) else {}
    payload["paths"] = _merge_paths(payload_paths if isinstance(payload_paths, dict) else None)
    payload["_config_path"] = str(cfg_path)
    return payload


def get_path(key: str, config: dict[str, Any] | None = None) -> Path:
    cfg = config if config is not None else load_config()
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    raw = paths.get(key, DEFAULT_PATHS.get(key, ""))
    return Path(str(raw))
