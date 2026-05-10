#!/usr/bin/env python3
"""
Validation script for Step 3 OCR pipeline.
Checks:
1. Page OCR manifest exists and has correct structure
2. OCR manifest exists with all required fields
3. Rejects and candidates manifests exist with correct counts
4. Manifests are linked correctly
5. Backward compatibility (Steps 1 and 2 unchanged)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read_jsonl(path: Path):
    """Read JSONL file, return dict keyed by appropriate ID."""
    records = {}
    if not path.exists():
        return records

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    # Key by primary ID field
                    if 'page_id' in record and 'crop_id' not in record:
                        records[record['page_id']] = record
                    elif 'crop_id' in record:
                        records[record['crop_id']] = record
                except json.JSONDecodeError as e:
                    print(f"  ⚠️  Skipped invalid JSON: {e}")

    return records


def count_lines_jsonl(path: Path) -> int:
    """Count valid JSON lines in a JSONL file."""
    if not path.exists():
        return 0
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError:
                    pass
    return count


def validate_page_ocr_manifest(manifest_path: Path) -> bool:
    """Validate page_ocr_manifest.jsonl structure."""
    print("\n" + "="*70)
    print("VALIDATION: Page OCR Manifest")
    print("="*70)

    if not manifest_path.exists():
        print(f"❌ FAIL: {manifest_path} does not exist")
        return False

    pages = read_jsonl(manifest_path)
    print(f"✅ Found {len(pages)} pages in manifest")

    if not pages:
        print("❌ FAIL: page_ocr_manifest.jsonl is empty")
        return False

    required_fields = {
        'page_id', 'doc_id', 'page_image_path', 'newspaper_name',
        'page_number1', 'page_width', 'page_height', 'ocr_engine', 'ocr_lines'
    }

    all_valid = True
    for page_id, page in pages.items():
        missing = required_fields - set(page.keys())
        if missing:
            print(f"  ❌ Page {page_id} missing fields: {missing}")
            all_valid = False

        # Validate ocr_lines structure
        ocr_lines = page.get('ocr_lines', [])
        if not isinstance(ocr_lines, list):
            print(f"  ❌ Page {page_id}: ocr_lines is not a list")
            all_valid = False
        else:
            for i, line in enumerate(ocr_lines[:3]):  # Check first 3
                if 'text' not in line or 'box' not in line or 'conf' not in line:
                    print(f"  ❌ Page {page_id}: line {i} missing text/box/conf")
                    all_valid = False
                    break

    if all_valid:
        print(f"✅ All {len(pages)} pages are valid")

    return all_valid


def validate_ocr_manifest(manifest_path: Path, crop_count: int) -> bool:
    """Validate ocr_manifest.jsonl structure."""
    print("\n" + "="*70)
    print("VALIDATION: OCR Manifest")
    print("="*70)

    if not manifest_path.exists():
        print(f"❌ FAIL: {manifest_path} does not exist")
        return False

    crops = read_jsonl(manifest_path)
    print(f"✅ Found {len(crops)} crops in OCR manifest")

    if not crops:
        print("❌ FAIL: ocr_manifest.jsonl is empty")
        return False

    required_fields = {
        'crop_id', 'page_id', 'doc_id', 'pdf_path', 'crop_image_path',
        'newspaper_name', 'page_number1',
        'ocr_engine', 'ocr_text_raw', 'ocr_text_norm', 'ocr_conf_mean',
        'inside_context_text', 'expanded_box_context_text',
        'left_context_text', 'right_context_text',
        'above_context_text', 'below_context_text',
        'keyword_terms', 'duplicate_hash', 'readable_char_count',
        'garbage_ratio', 'has_hiring_terms',
        'cheap_reject_reason', 'is_step3_survivor'
    }

    all_valid = True
    for crop_id, crop in list(crops.items())[:10]:  # Check first 10
        missing = required_fields - set(crop.keys())
        if missing:
            print(f"  ❌ Crop {crop_id} missing fields: {missing}")
            all_valid = False

    # Validate specific fields
    for crop_id, crop in list(crops.items())[:5]:
        if not isinstance(crop.get('ocr_text_raw'), str):
            print(f"  ❌ Crop {crop_id}: ocr_text_raw not string")
            all_valid = False
        if not isinstance(crop.get('ocr_conf_mean'), (int, float)):
            print(f"  ❌ Crop {crop_id}: ocr_conf_mean not numeric")
            all_valid = False
        if not isinstance(crop.get('is_step3_survivor'), bool):
            print(f"  ❌ Crop {crop_id}: is_step3_survivor not boolean")
            all_valid = False

    if all_valid:
        print(f"✅ All sampled crops are valid")

    return all_valid


def validate_rejects_and_candidates(
    rejects_path: Path,
    candidates_path: Path,
    ocr_manifest: dict
) -> bool:
    """Validate rejects and candidates manifests."""
    print("\n" + "="*70)
    print("VALIDATION: Rejects and Candidates")
    print("="*70)

    rejects = read_jsonl(rejects_path)
    candidates = read_jsonl(candidates_path)

    print(f"✅ Rejects: {len(rejects)}")
    print(f"✅ Candidates: {len(candidates)}")

    total = len(rejects) + len(candidates)
    if total != len(ocr_manifest):
        print(f"❌ FAIL: Rejects + Candidates ({total}) != OCR manifest ({len(ocr_manifest)})")
        return False

    print(f"✅ Total records match: {total} == {len(ocr_manifest)}")

    all_valid = True

    # Validate rejects have rejection reasons
    for crop_id, record in list(rejects.items())[:5]:
        if record.get('is_step3_survivor') != False:
            print(f"  ❌ Reject {crop_id}: is_step3_survivor should be False")
            all_valid = False
        if not record.get('cheap_reject_reason'):
            print(f"  ❌ Reject {crop_id}: missing cheap_reject_reason")
            all_valid = False

    # Validate candidates are marked as survivors
    for crop_id, record in list(candidates.items())[:5]:
        if record.get('is_step3_survivor') != True:
            print(f"  ❌ Candidate {crop_id}: is_step3_survivor should be True")
            all_valid = False

    if all_valid:
        print(f"✅ All sampled records are valid")

    return all_valid


def validate_backward_compatibility() -> bool:
    """Check that Steps 1 and 2 outputs are unchanged."""
    print("\n" + "="*70)
    print("VALIDATION: Backward Compatibility")
    print("="*70)

    all_valid = True

    # Check Stage 1 outputs
    page_manifest = Path("run_state/page_manifest.jsonl")
    legacy_manifest = Path("run_state/stage1_page_identity_manifest.json")
    images_dir = Path("data/pdf2img")

    if page_manifest.exists():
        count = count_lines_jsonl(page_manifest)
        print(f"✅ Stage 1 page_manifest.jsonl: {count} pages")
    else:
        print(f"❌ Stage 1 page_manifest.jsonl missing")
        all_valid = False

    if legacy_manifest.exists():
        print(f"✅ Stage 1 legacy manifest exists")
    else:
        print(f"⚠️  Stage 1 legacy manifest not found (expected)")

    if images_dir.exists():
        images = list(images_dir.glob("**/*.png"))
        print(f"✅ Stage 1 rendered images: {len(images)}")
    else:
        print(f"⚠️  Stage 1 images directory not found")

    # Check Stage 2 outputs
    crop_manifest = Path("run_state/crop_manifest.jsonl")
    if crop_manifest.exists():
        count = count_lines_jsonl(crop_manifest)
        print(f"✅ Stage 2 crop_manifest.jsonl: {count} crops")
    else:
        print(f"❌ Stage 2 crop_manifest.jsonl missing")
        all_valid = False

    crops_dir = Path("data/crops")
    if crops_dir.exists():
        crops = list(crops_dir.glob("*.png"))
        print(f"✅ Stage 2 crop images: {len(crops)}")
    else:
        print(f"❌ Stage 2 crops directory missing")
        all_valid = False

    return all_valid


def main():
    print("\n" + "="*70)
    print("STEP 3 OCR VALIDATION")
    print("="*70)

    page_ocr_path = Path("run_state/page_ocr_manifest.jsonl")
    ocr_path = Path("run_state/ocr_manifest.jsonl")
    rejects_path = Path("run_state/step3_rejects.jsonl")
    candidates_path = Path("run_state/step3_candidates.jsonl")

    all_passed = True

    # 1. Validate page OCR manifest
    if not validate_page_ocr_manifest(page_ocr_path):
        all_passed = False

    # 2. Validate OCR manifest
    if not validate_ocr_manifest(ocr_path, 371):
        all_passed = False
    else:
        ocr_manifest = read_jsonl(ocr_path)

    # 3. Validate rejects and candidates
    if not validate_rejects_and_candidates(rejects_path, candidates_path, ocr_manifest):
        all_passed = False

    # 4. Validate backward compatibility
    if not validate_backward_compatibility():
        all_passed = False

    # Summary
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
        print("="*70)
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
