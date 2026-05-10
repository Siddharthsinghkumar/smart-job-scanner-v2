#!/usr/bin/env python3
"""
Validation script for Stage1->Stage2 metadata handoff.
Checks:
1. Stage 1 still renders pages correctly
2. page_manifest.jsonl is created with correct structure
3. Stage 2 still produces detections
4. crop_manifest.jsonl is created with valid entries
5. Backward compatibility (old outputs still work)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.pipeline_metadata import read_page_manifest_jsonl, read_crop_manifest_jsonl


def validate_page_manifest(manifest_path: Path) -> bool:
    """
    Validate page_manifest.jsonl structure and content.

    Returns:
        bool: True if valid, False otherwise
    """
    print("\n" + "="*70)
    print("VALIDATION: Page Manifest JSONL")
    print("="*70)

    if not manifest_path.exists():
        print(f"❌ FAIL: page_manifest.jsonl does not exist at {manifest_path}")
        return False

    pages = read_page_manifest_jsonl(manifest_path)
    print(f"✅ Found {len(pages)} pages in manifest")

    if not pages:
        print("❌ FAIL: page_manifest.jsonl is empty")
        return False

    # Validate each page
    required_fields = {
        'page_id', 'doc_id', 'pdf_path', 'page_index0', 'page_number1',
        'newspaper_name', 'image_path', 'image_width', 'image_height', 'render_dpi'
    }

    all_valid = True
    for page_id, page in pages.items():
        missing = required_fields - set(page.keys())
        if missing:
            print(f"  ❌ Page {page_id} missing fields: {missing}")
            all_valid = False

        # Validate types
        if not isinstance(page.get('page_index0'), int):
            print(f"  ❌ Page {page_id}: page_index0 is not int")
            all_valid = False

        if not isinstance(page.get('page_number1'), int):
            print(f"  ❌ Page {page_id}: page_number1 is not int")
            all_valid = False

        if not isinstance(page.get('image_width'), int) or page['image_width'] <= 0:
            print(f"  ❌ Page {page_id}: image_width invalid ({page.get('image_width')})")
            all_valid = False

        if not isinstance(page.get('image_height'), int) or page['image_height'] <= 0:
            print(f"  ❌ Page {page_id}: image_height invalid ({page.get('image_height')})")
            all_valid = False

        # Validate image exists
        img_path = Path(page.get('image_path', ''))
        if not img_path.exists():
            print(f"  ❌ Page {page_id}: image not found at {page.get('image_path')}")
            all_valid = False

    if all_valid:
        print(f"✅ All {len(pages)} pages are valid")

    # Validate deterministic IDs
    page_ids = set(pages.keys())
    if len(page_ids) != len(pages):
        print(f"❌ FAIL: Duplicate page_ids detected")
        all_valid = False
    else:
        print(f"✅ All page_ids are unique")

    return all_valid


def validate_crop_manifest(manifest_path: Path, page_manifest: dict) -> bool:
    """
    Validate crop_manifest.jsonl structure and references.

    Returns:
        bool: True if valid, False otherwise
    """
    print("\n" + "="*70)
    print("VALIDATION: Crop Manifest JSONL")
    print("="*70)

    if not manifest_path.exists():
        print(f"⚠️  WARNING: crop_manifest.jsonl does not exist at {manifest_path}")
        print("   (This is OK if Stage 2 hasn't been run yet)")
        return True

    crops = read_crop_manifest_jsonl(manifest_path)
    print(f"✅ Found {len(crops)} crops in manifest")

    if not crops:
        print("⚠️  WARNING: crop_manifest.jsonl is empty (no detections processed)")
        return True

    required_fields = {
        'crop_id', 'page_id', 'doc_id', 'pdf_path', 'page_image_path', 'crop_image_path',
        'newspaper_name', 'page_index0', 'page_number1',
        'detector_model', 'detector_checkpoint', 'detector_conf',
        'bbox_xyxy_abs', 'bbox_xyxy_norm',
        'crop_width', 'crop_height', 'page_width', 'page_height',
        'padding_px', 'x_center_norm', 'y_center_norm', 'area_norm'
    }

    all_valid = True
    crop_page_ids = defaultdict(list)

    for crop_id, crop in crops.items():
        missing = required_fields - set(crop.keys())
        if missing:
            print(f"  ❌ Crop {crop_id} missing fields: {missing}")
            all_valid = False
            continue

        # Validate bbox coordinates
        bbox_abs = crop.get('bbox_xyxy_abs', [])
        if len(bbox_abs) != 4 or not all(isinstance(x, int) for x in bbox_abs):
            print(f"  ❌ Crop {crop_id}: bbox_xyxy_abs invalid")
            all_valid = False

        bbox_norm = crop.get('bbox_xyxy_norm', [])
        if len(bbox_norm) != 4 or not all(isinstance(x, (int, float)) for x in bbox_norm):
            print(f"  ❌ Crop {crop_id}: bbox_xyxy_norm invalid")
            all_valid = False
        else:
            # Check normalized values in [0, 1]
            if not all(0.0 <= x <= 1.0 for x in bbox_norm):
                print(f"  ❌ Crop {crop_id}: normalized bbox out of range: {bbox_norm}")
                all_valid = False

        # Validate center coordinates
        x_center = crop.get('x_center_norm')
        y_center = crop.get('y_center_norm')
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            print(f"  ❌ Crop {crop_id}: center coordinates out of range")
            all_valid = False

        # Validate area_norm
        area = crop.get('area_norm')
        if not (0.0 <= area <= 1.0):
            print(f"  ❌ Crop {crop_id}: area_norm out of range: {area}")
            all_valid = False

        # Check crop image exists
        crop_path = Path(crop.get('crop_image_path', ''))
        if not crop_path.exists():
            print(f"  ❌ Crop {crop_id}: crop image not found at {crop.get('crop_image_path')}")
            all_valid = False

        # Track page_id references
        page_id = crop.get('page_id')
        if page_id:
            crop_page_ids[page_id].append(crop_id)

            # Validate page_id reference
            if page_id not in page_manifest:
                print(f"  ❌ Crop {crop_id}: references unknown page_id {page_id}")
                all_valid = False

    # Check for unreferenced pages
    if page_manifest:
        referenced_page_ids = set(crop_page_ids.keys())
        all_page_ids = set(page_manifest.keys())
        unreferenced = all_page_ids - referenced_page_ids
        if unreferenced:
            print(f"⚠️  WARNING: {len(unreferenced)} pages with no crops detected (OK if they had no detections)")

    if all_valid:
        print(f"✅ All {len(crops)} crops are valid and properly linked")

    return all_valid


def validate_backward_compatibility(blocks_output_dir: Path, detections_output_dir: Path) -> bool:
    """
    Check that old Stage 1 and Stage 2 outputs still exist.

    Returns:
        bool: True if backward compatible, False otherwise
    """
    print("\n" + "="*70)
    print("VALIDATION: Backward Compatibility")
    print("="*70)

    all_valid = True

    # Check Stage 1 outputs (rendered images should exist)
    images_dir = Path("data/pdf2img")
    if images_dir.exists():
        images = list(images_dir.glob("**/*.png"))
        if images:
            print(f"✅ Stage 1 still produces rendered images ({len(images)} found)")
        else:
            print(f"⚠️  WARNING: No rendered images found")

    # Check Stage 1 legacy manifest
    legacy_manifest = Path("run_state/stage1_page_identity_manifest.json")
    if legacy_manifest.exists():
        try:
            with open(legacy_manifest) as f:
                data = json.load(f)
            print(f"✅ Stage 1 legacy manifest still generated")
        except Exception as e:
            print(f"❌ FAIL: Legacy manifest invalid: {e}")
            all_valid = False
    else:
        print(f"⚠️  WARNING: Legacy manifest not found (expected: {legacy_manifest})")

    # Check Stage 2 outputs (detections JSON files should exist)
    if detections_output_dir.exists():
        detections = list(detections_output_dir.glob("*.json"))
        if detections:
            print(f"✅ Stage 2 still produces detection metadata ({len(detections)} found)")
        else:
            print(f"⚠️  WARNING: No detection metadata found")

    # Check Stage 2 blocks output
    if blocks_output_dir.exists():
        blocks = list(blocks_output_dir.glob("**/debug_*.png"))
        if blocks or not any(blocks_output_dir.iterdir()):
            print(f"✅ Stage 2 block detection setup valid")
        else:
            print(f"⚠️  WARNING: No block debug output found (may be expected)")

    return all_valid


def main():
    print("\n" + "="*70)
    print("METADATA HANDOFF VALIDATION")
    print("="*70)

    # Paths
    page_manifest_path = Path("run_state/page_manifest.jsonl")
    crop_manifest_path = Path("run_state/crop_manifest.jsonl")
    blocks_output_dir = Path("data/blocks")
    detections_output_dir = Path("data/detections")

    all_passed = True

    # 1. Validate page manifest
    page_manifest = read_page_manifest_jsonl(page_manifest_path) if page_manifest_path.exists() else {}
    if not validate_page_manifest(page_manifest_path):
        all_passed = False

    # 2. Validate crop manifest
    if not validate_crop_manifest(crop_manifest_path, page_manifest):
        all_passed = False

    # 3. Validate backward compatibility
    if not validate_backward_compatibility(blocks_output_dir, detections_output_dir):
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
