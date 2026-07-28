#!/usr/bin/env python3
"""Image preprocessing facade used by OCR stage."""

from src.pipeline.stage04_ocr import preprocess_image  # noqa: F401

__all__ = ['preprocess_image']
