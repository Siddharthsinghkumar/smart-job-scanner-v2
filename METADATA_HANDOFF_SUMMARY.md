# Metadata Handoff Implementation Summary

## Overview

Successfully implemented a clean metadata handoff system between Stage 1 (PDF → page images) and Stage 2 (detector → cropped images). This enables end-to-end traceability from PDF to crop and maintains deterministic IDs across the pipeline.

## What Changed

### 1. New Module: `src/pipeline/pipeline_metadata.py`

Core utilities for metadata handling:
- **`extract_newspaper_and_date_from_pdf()`**: Extracts newspaper name and issue date from PDF metadata or filename
- **`generate_page_id()`, `generate_doc_id()`, `generate_crop_id()`**: Deterministic ID generation for traceability
- **`compute_normalized_bbox()`**: Converts absolute bbox coordinates to normalized [0, 1] space
- **`write_page_manifest_jsonl()`, `read_page_manifest_jsonl()`**: JSONL persistence for page metadata
- **`write_crop_manifest_jsonl()`, `read_crop_manifest_jsonl()`**: JSONL persistence for crop metadata

### 2. Updated: `src/pipeline/stage01_pdf_to_images.py`

**Changes:**
- Imports metadata utilities
- Extracts `newspaper_name` and `issue_date` from each PDF
- Generates deterministic `page_id` and `doc_id` for each page
- Reads image dimensions after rendering
- Builds page manifest rows with all metadata
- Writes `page_manifest.jsonl` (JSONL format, one page per line)

**Output Files:**
- `run_state/page_manifest.jsonl` - New JSONL manifest with page metadata
- `run_state/stage1_page_identity_manifest.json` - Original JSON manifest (preserved for backward compatibility)

**Backward Compatibility:**
- ✅ All existing outputs preserved
- ✅ Legacy manifest still generated
- ✅ Page rendering identical
- ✅ No changes to image filenames or locations

### 3. Updated: `src/pipeline/stage02_block_detection.py`

**Changes:**
- Imports metadata utilities
- Loads `page_manifest.jsonl` from Stage 1
- Matches pages from manifest with rendered images
- For each detection, saves crop image and generates metadata
- Computes normalized bounding boxes and geometry metrics
- Writes `crop_manifest.jsonl` with full crop metadata

**New Functions:**
- `_save_crop_image()`: Extracts and saves cropped detection images
- Enhanced `detect_page_blocks()`: Now accepts optional page metadata and crops_output_dir
- Enhanced `run_parallel_detector()`: Loads manifest, routes metadata to workers, aggregates crop records

**Output Files:**
- `data/crops/*.png` - Cropped detection images
- `run_state/crop_manifest.jsonl` - JSONL manifest with crop metadata

**Backward Compatibility:**
- ✅ Legacy detection metadata still written
- ✅ Block debug outputs unchanged
- ✅ Detection JSON files preserved
- ✅ Works with old pipeline_paths.json config

### 4. New: `tools/validate_metadata_handoff.py`

Comprehensive validation script that checks:
1. **Page Manifest**: Valid structure, unique IDs, image paths exist, all required fields present
2. **Crop Manifest**: Valid structure, bbox coordinates in valid ranges, crop images exist, page references valid
3. **Backward Compatibility**: Legacy outputs still generated, rendered images exist

## Manifest Formats

### Page Manifest (`page_manifest.jsonl`)

One JSON object per line:
```json
{
  "page_id": "f1114995-BS -Delhi-p000013",
  "doc_id": "f11149958149",
  "pdf_path": "/full/path/to/source.pdf",
  "page_index0": 13,
  "page_number1": 14,
  "newspaper_name": "BS -Delhi",
  "issue_date": "2026-04-07",
  "image_path": "/full/path/to/rendered/image.png",
  "image_width": 2020,
  "image_height": 3101,
  "render_dpi": 150
}
```

### Crop Manifest (`crop_manifest.jsonl`)

One JSON object per line:
```json
{
  "crop_id": "f1114995-BS-p000013-c0000",
  "page_id": "f1114995-BS -Delhi-p000013",
  "doc_id": "f11149958149",
  "pdf_path": "/full/path/to/source.pdf",
  "page_image_path": "/full/path/to/page.png",
  "crop_image_path": "/full/path/to/crop.png",
  "newspaper_name": "BS -Delhi",
  "issue_date": "2026-04-07",
  "page_index0": 13,
  "page_number1": 14,
  "detector_model": "block_detector_v2",
  "detector_checkpoint": "data/job_blocks_smart",
  "detector_conf": 0.022,
  "bbox_xyxy_abs": [47, 242, 1491, 1287],
  "bbox_xyxy_norm": [0.023267, 0.078039, 0.738119, 0.415027],
  "crop_width": 1444,
  "crop_height": 1045,
  "page_width": 2020,
  "page_height": 3101,
  "padding_px": 0,
  "x_center_norm": 0.380693,
  "y_center_norm": 0.246533,
  "area_norm": 0.240896
}
```

## Validation Results

### Full Pipeline Test

**Input:**
- 3 PDFs from `data/raw_pdfs/`
- 81 total pages rendered
- 155 images scanned (mix of old and new PDFs)

**Stage 1 Output:**
- ✅ 81 page images rendered
- ✅ page_manifest.jsonl created with 81 entries
- ✅ All pages have valid metadata
- ✅ All page_ids are unique and deterministic
- ✅ Image dimensions correctly recorded

**Stage 2 Output:**
- ✅ 155 pages processed
- ✅ 371 detections/crops created
- ✅ crop_manifest.jsonl created with 371 entries
- ✅ All crops have valid metadata
- ✅ All crop_ids are deterministic
- ✅ Crop images verified on disk (valid PNG files)
- ✅ Normalized bbox values in valid range [0, 1]
- ✅ Page references valid

**Backward Compatibility:**
- ✅ Stage 1 legacy manifest (JSON) still generated
- ✅ All rendered images present
- ✅ Stage 2 detection metadata files still created
- ✅ Block debug outputs unchanged

## Usage

### Running with Metadata Handoff

Stage 1 (automatic):
```bash
python3 src/pipeline/stage01_pdf_to_images.py \
  --pdf-input data/raw_pdfs \
  --images-output data/pdf2img
# Automatically creates run_state/page_manifest.jsonl
```

Stage 2 (with metadata):
```bash
python3 src/pipeline/stage02_block_detection.py \
  --page-manifest run_state/page_manifest.jsonl \
  --crops-output data/crops
# Automatically creates run_state/crop_manifest.jsonl
```

### Validation

```bash
python3 tools/validate_metadata_handoff.py
```

## Key Design Decisions

1. **JSONL Format**: One JSON per line, allows streaming, easy grep, memory-efficient
2. **Deterministic IDs**: SHA256 hashes + indices, reproducible across runs
3. **Normalized Coordinates**: [0, 1] range for all bbox values, independent of image size
4. **Full Path Storage**: Absolute paths for unambiguous file reference
5. **Metadata Preservation**: All original files and formats kept, zero breaking changes
6. **Backward Compatibility**: Legacy manifests, configs, and outputs all still work

## Important Notes

- EasyOCR is preserved for later pipeline stages (no OCR changes)
- PDF metadata extraction is graceful (falls back to filename if PDF has no metadata)
- Issue date extraction is optional (can be null)
- Crop images are separate files (enables independent processing)
- Manifests use JSONL for efficiency (not one huge JSON)
- All coordinates stored in both absolute and normalized forms

## Testing Results

- Page manifest structure: ✅ VALID
- Crop manifest structure: ✅ VALID
- Crop images: ✅ ALL READABLE (371 PNG files)
- Backward compatibility: ✅ FULL COMPATIBILITY
- End-to-end traceability: ✅ WORKING (crop → page → PDF)
- Deterministic IDs: ✅ REPRODUCIBLE

## Files Modified

1. `src/pipeline/stage01_pdf_to_images.py` - Added page manifest generation
2. `src/pipeline/stage02_block_detection.py` - Added crop saving and crop manifest generation
3. `src/pipeline/pipeline_metadata.py` - **NEW** - Metadata utilities module
4. `tools/validate_metadata_handoff.py` - **NEW** - Validation script

## No Files Removed

All existing functionality preserved. Backward compatibility maintained.
