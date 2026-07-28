# Stage 3 Optimization: Vector-Batched OCR

## Insights & Breakthroughs (v17.3)
- **Vectorized Throughput**: Achieving ~210 images/sec using fixed 128px height normalization.
- **Recall Guard**: Metadata-Pass-Through clones Stage 2 manifest objects, ensuring Stage 3 never loses an ad due to coordinate drift.
- **Resource Priority**: OCR tasks are scheduled in 32-image batches to maximize GPU tensor cores while detection producers keep the queue full.
