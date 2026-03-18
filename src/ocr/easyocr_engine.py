#!/usr/bin/env python3
"""Unified OCR engine facade for production pipeline."""

from src.pipeline.stage04_ocr import extract_all_folders, reset_progress_and_outputs  # noqa: F401

__all__ = ['extract_all_folders', 'reset_progress_and_outputs']
