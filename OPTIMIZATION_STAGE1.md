# Stage 1 Optimization: Accelerated Rendering

## Insights & Breakthroughs (v17.0)
- **Library Isolation**: Confirmed that unsetting `LD_LIBRARY_PATH` is required on this system to prevent Torch from conflicting with system CUDA 12.6 libraries.
- **Fast Path Persistence**: Digital pages (determined by character density) bypass Stage 2 entirely, saving ~90% of processing time for modern PDFs.
- **Rendering Speed**: Strictly hitting ~200ms per page at 300 DPI using `fitz` memory-stream.
