# Monitoring Learnings & Fixes

## Failure Audit (v17.6)
- **Problem**: Cascading NameErrors and deadlocks during high-concurrency 10-PDF runs.
- **Root Cause**: Global library imports in parent process causing CUDA context lock-up in 'spawn' children.
- **Solution**: **Warm Engine v16.12** uses Lazy-Loading, 1 GPU Worker (for 4GB VRAM), and Explicit Queue Draining.

## The Final Monitor Protocol (v17.7)
1. **Lazy Loading**: NEVER import `torch` or `ultralytics` at the module level when using `spawn`.
2. **GPU Singleton**: Limit to 1 GPU worker for 4GB VRAM to avoid context-switching overhead.
3. **Queue Safety**: Always use `queue.Empty` and ensure all task queues are drained before worker exit.
4. **Sentry Sensitivity**: 15% CPU / 45s inactivity thresholds capture stalls without killing slow OCR batches.

## 🚩 Major Violation Incident (May 7, 2026)
- **Incident**: 1-hour stall with 5 consecutive command timeouts.
- **Root Cause**: Agent violated **Lazy Loading Mandate**. `torch` and `easyocr` were imported at the top-level of `scripts/unified_bench_v2.py`.
- **Result**: Immediate CUDA context deadlock upon process start, leading to "CUDA unknown error" and subsequent hardware instability (749W sensor anomaly).
- **Corrective Action**: Implemented "Fatal Hardware Error Protocol" in `GEMINI.md`.

