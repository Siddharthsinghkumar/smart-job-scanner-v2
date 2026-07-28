#!/usr/bin/env python3
"""
Step 3 OCR utilities: text normalization, context extraction, rejection logic.
"""

import re
import hashlib
from typing import Optional, Dict, List, Tuple
import unicodedata


def normalize_text(text: str) -> str:
    """
    Normalize OCR text: lowercase, remove extra whitespace, normalize Unicode.
    Preserves structure but makes comparable.
    """
    if not text:
        return ""

    # Normalize Unicode
    text = unicodedata.normalize('NFKD', text)

    # Lowercase
    text = text.lower()

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Trim
    text = text.strip()

    return text


def compute_duplicate_hash(text: str) -> str:
    """Compute deterministic hash of normalized text for duplicate detection."""
    normalized = normalize_text(text)
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


def count_readable_chars(text: str) -> int:
    """Count alphanumeric + punctuation characters (exclude whitespace)."""
    if not text:
        return 0
    return sum(1 for c in text if c.isalnum() or c in '.,!?;:-\'"()')


def compute_garbage_ratio(text: str) -> float:
    """
    Estimate ratio of 'garbage' characters (control chars, symbols, etc.)
    to readable content.

    Returns: float in [0, 1] where 0 = no garbage, 1 = all garbage
    """
    if not text:
        return 0.0

    readable = count_readable_chars(text)
    garbage = len(text) - readable

    if len(text) == 0:
        return 1.0

    return garbage / len(text)


def has_hiring_language(text: str) -> bool:
    """
    Check if text contains job/hiring-related keywords.
    Cheap heuristic to identify job ads vs random text.
    """
    if not text:
        return False

    normalized = normalize_text(text)

    hiring_keywords = {
        'job', 'jobs', 'hire', 'hiring', 'recruit', 'vacancy', 'vacancies',
        'position', 'positions', 'apply', 'application', 'candidate',
        'experience', 'skill', 'skills', 'salary', 'salary', 'pay',
        'company', 'organization', 'team', 'department',
        'work', 'working', 'workplace',
        'ad', 'advertisement', 'advertise',
        'interview', 'requirement', 'requirements',
        'desired', 'required',
        'role', 'roles',
    }

    words = set(normalized.split())
    return bool(words & hiring_keywords)


def extract_context_from_page_ocr(
    crop_bbox_norm: Tuple[float, float, float, float],
    page_ocr_lines: List[Dict],
    direction: str,
    expand_ratio: float = 0.2
) -> str:
    """
    Extract context text from page OCR near a crop's bounding box.

    Args:
        crop_bbox_norm: (x1, y1, x2, y2) in [0, 1]
        page_ocr_lines: List of {text, box: {x1, y1, x2, y2}, conf}
        direction: 'inside', 'expanded', 'left', 'right', 'above', 'below'
        expand_ratio: How much to expand beyond crop (for 'expanded')

    Returns:
        Context text string
    """
    if not page_ocr_lines:
        return ""

    cx1, cy1, cx2, cy2 = crop_bbox_norm
    crop_width = cx2 - cx1
    crop_height = cy2 - cy1

    # Define search regions based on direction
    if direction == 'inside':
        # Lines entirely within crop
        x_min, y_min = cx1, cy1
        x_max, y_max = cx2, cy2
    elif direction == 'expanded':
        # Expanded crop region
        expand_x = crop_width * expand_ratio
        expand_y = crop_height * expand_ratio
        x_min = max(0, cx1 - expand_x)
        y_min = max(0, cy1 - expand_y)
        x_max = min(1, cx2 + expand_x)
        y_max = min(1, cy2 + expand_y)
    elif direction == 'left':
        # Lines to the left of crop
        x_min, x_max = 0, cx1
        y_min, y_max = cy1, cy2
    elif direction == 'right':
        # Lines to the right of crop
        x_min, x_max = cx2, 1
        y_min, y_max = cy1, cy2
    elif direction == 'above':
        # Lines above crop
        x_min, x_max = 0, 1
        y_min, y_max = 0, cy1
    elif direction == 'below':
        # Lines below crop
        x_min, x_max = 0, 1
        y_min, y_max = cy2, 1
    else:
        return ""

    matching_texts = []
    for line in page_ocr_lines:
        box = line.get('box', {})
        lx1 = box.get('x1', 0)
        ly1 = box.get('y1', 0)
        lx2 = box.get('x2', 1)
        ly2 = box.get('y2', 1)

        # Check if line overlaps with region
        if (lx2 >= x_min and lx1 <= x_max and
            ly2 >= y_min and ly1 <= y_max):
            text = line.get('text', '').strip()
            if text:
                matching_texts.append(text)

    return ' '.join(matching_texts)


class CheapRejector:
    """Simple, configurable rejection logic for obvious junk."""

    def __init__(
        self,
        min_text_length: int = 5,
        max_garbage_ratio: float = 0.6,
        min_ocr_confidence: float = 0.3,
        require_hiring_language: bool = False,
    ):
        """
        Args:
            min_text_length: Reject if readable chars < this
            max_garbage_ratio: Reject if garbage_ratio > this
            min_ocr_confidence: Reject if ocr_conf_mean < this
            require_hiring_language: If True, reject if no hiring keywords found
        """
        self.min_text_length = min_text_length
        self.max_garbage_ratio = max_garbage_ratio
        self.min_ocr_confidence = min_ocr_confidence
        self.require_hiring_language = require_hiring_language

    def evaluate(self, ocr_record: Dict) -> Tuple[bool, Optional[str]]:
        """
        Evaluate if record is a survivor (True) or should be rejected (False).

        Returns:
            (is_survivor, rejection_reason_or_none)
        """
        text_raw = ocr_record.get('ocr_text_raw', '')
        ocr_conf = ocr_record.get('ocr_conf_mean', 0.0)

        # Check OCR confidence
        if ocr_conf < self.min_ocr_confidence:
            return False, f"low_ocr_conf:{ocr_conf:.2f}"

        # Check text length
        readable_count = count_readable_chars(text_raw)
        if readable_count < self.min_text_length:
            return False, f"too_short:{readable_count}"

        # Check garbage ratio
        garbage = compute_garbage_ratio(text_raw)
        if garbage > self.max_garbage_ratio:
            return False, f"too_garbage:{garbage:.2f}"

        # Check for hiring language (if required)
        if self.require_hiring_language:
            if not has_hiring_language(text_raw):
                return False, "no_hiring_language"

        return True, None
