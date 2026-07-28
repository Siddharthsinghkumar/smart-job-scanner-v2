# Stage 2 Optimization: Nitro-Parallel Detection

## Insights & Breakthroughs (v17.3)
- **VRAM Bound**: Confirmed that 4 GPU workers saturation the 4GB RTX 3050 Ti at 98%. Attempting 5+ workers results in kernel deadlocks.
- **Recall Key**: 0.52 overlap and 0.0005 confidence provides the optimal balance of 100% recall and manageable candidate volume.
- **Persistence Reset**: Identified that low-level hardware stalls require a "Cold Boot" or `nvidia-smi --gpu-reset` to clear the corrupted kernel state.

## Final Baseline
- **Speed**: ~55s for full UHT Delhi (26 pages).
- **Recall**: 100% (49/49 TP).
