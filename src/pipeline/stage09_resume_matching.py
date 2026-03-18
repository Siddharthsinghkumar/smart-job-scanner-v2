#!/usr/bin/env python3
"""Composite stage that preserves historical Stage-9 behavior.

Runs the original Stage-9 sub-steps in the same order:
1) dynamic resume generation
2) local similarity filter
3) cloud LLM filtering
4) shortlist generation
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    base = Path(__file__).resolve().parent
    steps = [
        base / 'stage09_dynamic_resumes.py',
        base / 'stage09_local_filter.py',
        base / 'stage09_llm_filter.py',
        base / 'stage09_shortlist.py',
    ]

    for step in steps:
        cmd = [sys.executable, str(step)]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            return rc
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
