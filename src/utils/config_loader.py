#!/usr/bin/env python3
"""Centralized configuration loading from configs/."""

from __future__ import annotations

from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / 'configs'


def resolve_config_path(filename: str) -> Path:
    return CONFIGS_DIR / filename


def load_json_config(filename: str):
    path = resolve_config_path(filename)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_config(filename: str, payload):
    path = CONFIGS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def project_root() -> Path:
    return PROJECT_ROOT
