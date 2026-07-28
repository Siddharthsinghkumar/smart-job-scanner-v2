#!/usr/bin/env python3
import json
from pathlib import Path


def load_state(path: Path, default=None):
    if default is None:
        default = {}
    if not path.exists():
        return default
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_state(path: Path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
