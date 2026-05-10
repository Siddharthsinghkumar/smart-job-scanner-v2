#!/usr/bin/env python3
"""
Final validation for merged Step 3 OCR pipeline.
Checks all outputs are correct and pipeline runs successfully.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def count_valid_jsonl(path: Path) -> int:
    """Count valid JSON lines."""
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


def load_jsonl(path: Path) -> dict:
    """Load JSONL file into dict."""
    records = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    rec = json.loads(line)
                    key = rec.get('crop_id') or rec.get('page_id')
                    if key:
                        records[key] = rec
                except json.JSONDecodeError:
                    pass
    return records


def main():
    print("\n" + "="*70)
    print("FINAL VALIDATION: Step 3 OCR (Merged Optimized)")
    print("="*70)

    ocr_path = Path("run_state/ocr_manifest.jsonl")
    candidates_path = Path("run_state/step3_candidates.jsonl")
    rejects_path = Path("run_state/step3_rejects.jsonl")
    page_ocr_path = Path("run_state/page_ocr_manifest.jsonl")

    all_pass = True

    # 1. Check all files exist
    print("\n1. OUTPUT FILES")
    print("-" * 70)
    for name, path in [
        ("OCR Manifest", ocr_path),
        ("Candidates", candidates_path),
        ("Rejects", rejects_path),
        ("Page OCR Cache", page_ocr_path),
    ]:
        if path.exists():
            count = count_valid_jsonl(path)
            size = path.stat().st_size / 1024 / 1024
            print(f"✅ {name:20s}: {count:4d} records ({size:.1f} MB)")
        else:
            print(f"❌ {name:20s}: MISSING")
            all_pass = False

    # 2. Check record counts
    print("\n2. RECORD COUNTS")
    print("-" * 70)
    ocr_count = count_valid_jsonl(ocr_path)
    candidates_count = count_valid_jsonl(candidates_path)
    rejects_count = count_valid_jsonl(rejects_path)
    pages_count = count_valid_jsonl(page_ocr_path)

    print(f"OCR manifest:  {ocr_count} crops")
    print(f"Candidates:    {candidates_count} survivors")
    print(f"Rejects:       {rejects_count} rejected")
    print(f"Page cache:    {pages_count} unique pages")

    if candidates_count + rejects_count == ocr_count:
        print(f"✅ Splits correct: {candidates_count} + {rejects_count} = {ocr_count}")
    else:
        print(f"❌ Split mismatch: {candidates_count} + {rejects_count} ≠ {ocr_count}")
        all_pass = False

    # 3. Check record structure
    print("\n3. RECORD STRUCTURE")
    print("-" * 70)

    required_fields = {
        'crop_id', 'page_id', 'doc_id', 'ocr_text_raw', 'ocr_text_norm',
        'ocr_conf_mean', 'inside_context_text', 'expanded_box_context_text',
        'left_context_text', 'right_context_text', 'above_context_text',
        'below_context_text', 'garbage_ratio', 'has_hiring_terms',
        'cheap_reject_reason', 'is_step3_survivor'
    }

    ocr_records = load_jsonl(ocr_path)
    sample_records = list(ocr_records.values())[:5]

    struct_ok = True
    for i, record in enumerate(sample_records):
        missing = required_fields - set(record.keys())
        if missing:
            print(f"❌ Sample {i}: missing {missing}")
            struct_ok = False

    if struct_ok:
        print(f"✅ All sampled records have required fields")
    else:
        all_pass = False

    # 4. Check data types
    print("\n4. DATA TYPE VALIDATION")
    print("-" * 70)

    types_ok = True
    for i, record in enumerate(sample_records[:3]):
        if not isinstance(record.get('is_step3_survivor'), bool):
            print(f"❌ is_step3_survivor not bool: {record.get('is_step3_survivor')}")
            types_ok = False
        if not isinstance(record.get('ocr_conf_mean'), (int, float)):
            print(f"❌ ocr_conf_mean not numeric: {record.get('ocr_conf_mean')}")
            types_ok = False
        if not isinstance(record.get('garbage_ratio'), (int, float)):
            print(f"❌ garbage_ratio not numeric: {record.get('garbage_ratio')}")
            types_ok = False

    if types_ok:
        print(f"✅ Data types are correct")
    else:
        all_pass = False

    # 5. Check rejection logic
    print("\n5. REJECTION LOGIC")
    print("-" * 70)

    reject_records = load_jsonl(rejects_path)
    sample_rejects = list(reject_records.values())[:5]

    reject_logic_ok = True
    for record in sample_rejects:
        if record.get('is_step3_survivor') != False:
            print(f"❌ Reject record marked as survivor: {record['crop_id']}")
            reject_logic_ok = False
        if not record.get('cheap_reject_reason'):
            print(f"❌ Reject record missing reason: {record['crop_id']}")
            reject_logic_ok = False

    if reject_logic_ok:
        print(f"✅ All rejects properly marked with reasons")
    else:
        all_pass = False

    # 6. Check context extraction
    print("\n6. CONTEXT EXTRACTION")
    print("-" * 70)

    context_fields = [
        'inside_context_text', 'expanded_box_context_text',
        'left_context_text', 'right_context_text',
        'above_context_text', 'below_context_text'
    ]

    context_ok = True
    for record in sample_records[:3]:
        for field in context_fields:
            if field not in record:
                print(f"❌ Missing context field: {field}")
                context_ok = False

    if context_ok:
        print(f"✅ All context fields present")
    else:
        all_pass = False

    # 7. Backward compatibility check
    print("\n7. BACKWARD COMPATIBILITY")
    print("-" * 70)

    # Check Step 1/2 outputs still exist
    page_manifest = Path("run_state/page_manifest.jsonl")
    crop_manifest = Path("run_state/crop_manifest.jsonl")
    rendered_images = list(Path("data/pdf2img").glob("**/*.png"))

    if page_manifest.exists():
        pm_count = count_valid_jsonl(page_manifest)
        print(f"✅ Page manifest: {pm_count} pages")
    else:
        print(f"❌ Page manifest missing")
        all_pass = False

    if crop_manifest.exists():
        cm_count = count_valid_jsonl(crop_manifest)
        print(f"✅ Crop manifest: {cm_count} crops")
    else:
        print(f"❌ Crop manifest missing")
        all_pass = False

    if rendered_images:
        print(f"✅ Rendered images: {len(rendered_images)} files")
    else:
        print(f"❌ No rendered images")
        all_pass = False

    # Summary
    print("\n" + "="*70)
    if all_pass:
        print("✅ ALL VALIDATIONS PASSED")
        print(f"\nSummary:")
        print(f"  - {ocr_count} crops OCR'd successfully")
        print(f"  - {candidates_count} survivors for downstream")
        print(f"  - {rejects_count} cheap rejections")
        print(f"  - {pages_count} unique pages cached")
        print("="*70)
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
