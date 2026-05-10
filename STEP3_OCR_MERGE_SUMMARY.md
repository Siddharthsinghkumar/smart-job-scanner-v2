# Step 3 OCR Pipeline: Merge & Optimization Summary

## Problem Statement

The original `stage03_ocr.py` (16 KB, 474 lines) was approximately **5x slower** than the legacy `stage04_ocr.py` (70 KB, 1589 lines). Despite having simpler code, the new version had a critical bottleneck: **sequential, synchronous OCR processing** with no worker parallelism.

## Root Cause Analysis

### Original stage03_ocr.py (Slow)
- **Sequential processing**: OCRs one crop at a time
- **Blocking I/O**: No async result writing - GPU waits while results write to disk
- **No page caching**: OCRs entire page for every crop that references it (wasteful)
- **No progress tracking**: Cannot resume interrupted runs
- **Single reader**: No GPU/CPU load balancing
- **Memory inefficiency**: Stores entire page OCR lines in every crop record

### Legacy stage04_ocr.py (Fast)
- **Worker queue architecture**: GPU + CPU workers process tasks in parallel
- **Async result writing**: Dedicated result writer thread prevents GPU idle time
- **Page-level caching**: Each page OCR'd once, results reused for multiple crops
- **Progress tracking**: JSON state files enable resume capability
- **GPU/CPU overlap**: CPU workers preprocess while GPU processes previous batch
- **OOM handling**: Intelligent spillover from GPU to CPU when memory tight
- **Image preprocessing cache**: Cached resized images avoid repeat preprocessing

## Merge Strategy

Combine the **architectural strengths of stage04** with the **manifest-based interfaces of stage03**:

1. ✅ **Keep from stage03**: Manifest-based pipeline (crop_manifest JSONL input), context extraction, cheap rejection
2. ✅ **Keep from stage04**: Worker queues, async result writing, page caching, progress tracking
3. ✅ **Adapt for manifests**: Instead of file system folder crawling, consume from loaded crop_manifest dict

## What Changed in Merged Version

### Architecture
- **Worker threads** (GPU + CPU workers) instead of sequential processing
- **Result queue** for async writing (prevents GPU blocking)
- **Page OCR cache** (thread-safe with lock) - one per page_id
- **Progress tracking** with JSON checkpoint files

### Performance Optimizations
- **GPU stays busy**: Result writing happens on dedicated thread
- **Parallel crop OCR**: GPU workers and CPU workers process crops simultaneously
- **CPU/GPU overlap**: Multiple crops queued while GPU processes one
- **Reduced memory**: No unnecessary duplication of page lines per crop

### Maintained Compatibility
- **Same manifest inputs**: crop_manifest.jsonl from Stage 2
- **Same manifest outputs**: page_ocr_manifest.jsonl, ocr_manifest.jsonl, step3_rejects.jsonl, step3_candidates.jsonl
- **Same fields**: All OCR record fields preserved
- **Same rejection logic**: CheapRejector with identical thresholds
- **Same context extraction**: Spatial region-based context (inside/expanded/left/right/above/below)

## Key Code Changes

### 1. Worker Architecture
```python
# Workers pull from queue, produce results
def crop_ocr_worker(task_queue, result_queue, crops_manifest, crops_dir, worker_id, gpu=True):
    while True:
        crop_id = task_queue.get()
        if crop_id is None: break
        # OCR and produce results
        result_queue.put((crop_id, ocr_record, "ok"))
```

### 2. Async Result Writing
```python
# Dedicated thread for writing (prevents GPU blocking)
def result_writer_worker(result_queue, output_paths):
    while True:
        crop_id, ocr_record, status = result_queue.get()
        # Write to appropriate file (ocr_manifest, rejects, candidates)
```

### 3. Page OCR Caching (Thread-Safe)
```python
PAGE_CACHE_LOCK = threading.Lock()
PAGE_CACHE = {}

def get_or_build_page_cache(page_id, crop, crops_dir):
    with locked(PAGE_CACHE_LOCK):
        if page_id in PAGE_CACHE:
            return PAGE_CACHE[page_id]
    # Build and cache
    with locked(PAGE_CACHE_LOCK):
        PAGE_CACHE[page_id] = page_ocr_record
```

### 4. Progress Tracking
```python
def save_progress(data):
    p = PROGRESS_DIR / "ocr_progress.json"
    atomic_write_json(p, data)

# Resume-aware: skip already-completed crops
prog = load_progress()
completed_crops = set(prog.keys())
pending_crops = [cid for cid in crop_ids if cid not in completed_crops]
```

## Configuration

```python
GPU_WORKERS = 1          # Single GPU worker (EasyOCR uses one CUDA context)
CPU_WORKERS = 2          # 2 CPU workers for preprocessing + fallback
GPU_TASK_QUEUE_SIZE = 512
RESULT_QUEUE_SIZE = 512
```

## Expected Performance Improvements

### Before (Sequential)
- 371 crops × ~0.5s per crop = ~185s minimum
- Actual observed: ~1 hour (slow)
- GPU utilization: Low (blocking waits)
- CPU utilization: Low (single-threaded)

### After (Queued Workers)
- 371 crops with GPU+CPU workers running in parallel
- Expected: ~15-20 minutes (3-4x faster)
- GPU utilization: High (no blocking I/O)
- CPU utilization: Better load spreading

## Testing Approach

1. **Run merged version** on full 371-crop sample
2. **Measure throughput**: crops/second
3. **Check output correctness**: Validate manifest structures
4. **Verify backward compatibility**: Same output fields and format
5. **Compare with old if possible**: Before/after timing

## Files Modified

- `src/pipeline/stage03_ocr.py` - Complete rewrite with merged architecture
- `src/pipeline/step3_ocr_utilities.py` - Unchanged (reused as-is)

## Files NOT Modified

- `src/pipeline/pipeline_metadata.py` - Input manifest reading
- `src/pipeline/stage01_pdf_to_images.py` - Upstream step
- `src/pipeline/stage02_block_detection.py` - Upstream step
- All test/validation scripts - Compatible with new output

## Known Limitations

1. **Thread-based, not process-based**: CPU workers run in same process (GIL impact minimal due to EasyOCR C++ backend)
2. **Single GPU**: Architecture assumes single GPU (no multi-GPU support)
3. **No distributed**: All workers run on same machine

## Future Optimizations (Not Included)

1. Image preprocessing cache (disk-based) - could save ~20-30% time
2. Multi-GPU support
3. Batched OCR (send multiple crops to EasyOCR at once)
4. Adaptive queue sizes based on VRAM usage

---

**Status**: ✅ Merged, ⏳ Testing in progress
