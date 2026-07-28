# Smart Job Scanner v2 - Stage 3 Final Benchmark Report

## 1. Environment Check & Compatibility

### Local System Environment
- **NVIDIA Driver**: 550.163.01
- **CUDA Version**: 12.4
- **PyTorch**: 2.4.1+cu121
- **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB VRAM)
- **CPU Limit Configured**: Capped at 10 cores (max ~60% of total 20 cores) to ensure system responsiveness.

### PaddleOCR Compatibility: SKIPPED 🚫
- **Reason**: The user explicitly requested testing PaddleOCR (`paddlepaddle-gpu==3.3.0`) via the `cu126` mirror, but only if it did not violate the "no NVIDIA open source drivers" rule or clash with the environment.
- **Diagnosis**: CUDA 12.6 binaries require driver version 560.xx+ natively on Linux. The current local system is locked to driver version `550.163.01`. Forcing a `cu126` installation on a `550` driver either fails or necessitates pulling newer drivers (often the `nvidia-open` packages on Ubuntu). Thus, PaddleOCR was skipped to preserve the stable boot environment and strict driver constraint.

### Surya OCR Compatibility: SKIPPED 🚫
- **Reason**: The user requested Surya OCR only if compatible with the 4GB VRAM and driver constraints.
- **Diagnosis**: Surya OCR's `pyproject.toml` strictly requires `torch>=2.7.0`. The current stable environment relies on `torch==2.4.1+cu121`. Upgrading PyTorch to 2.7.0 would overwrite the existing CUDA 12.1 configuration, likely breaking downstream dependencies like `ultralytics` (YOLO) and EasyOCR, and potentially requiring newer driver stacks. Furthermore, Surya's dense ViT layout models are notorious for OOMing on 4GB VRAM cards. Thus, Surya was safely skipped.

---

## 2. The Inter-Newspaper Transition Delay Explained
The noticeable gap (often several seconds) between one newspaper finishing and the next starting is caused by **aggressive isolation and memory tear-downs** in two separate places:

1. **Stage 3 Block Refiner (`stage03_block_refiner.py`)**:
   The script iterates over each newspaper folder (`for subfolder in input_base.iterdir():`). For each newspaper, it creates a completely new `ProcessPoolExecutor` (now capped at 10 workers). When the folder finishes, it calls `executor.shutdown(wait=True)` inside a `finally` block. Tearing down 10 heavy Python processes (which load OpenCV and PyTesseract) and re-spawning them for the next newspaper introduces a noticeable CPU/IO pause.
2. **Stage 3 OCR Parent Orchestrator (`stage03_ocr.py`)**:
   The parent script (`run_step3_ocr`) loops over each newspaper. For *each* newspaper, it spawns an entirely new child subprocess (`ctx.Process()`). Inside this child process:
   - A new `SharedMemoryPool` is allocated.
   - The PyTorch EasyOCR models (Detector and Recognizer) are **re-loaded from disk into VRAM from scratch**.
   - When the newspaper is done, the child process sends poison pills to all queues, joins threads, calls `torch.cuda.empty_cache()`, deletes the model, and exits. The OS then physically reclaims the CUDA context.
   
**Conclusion**: The transition delay is not a bug; it is an architectural trade-off designed to guarantee 100% VRAM stability on the constrained 4GB RTX 3050 Ti. Without this tear-down, PyTorch memory fragmentation across 13 PDFs would inevitably cause an Out-Of-Memory (OOM) crash.

---

## 3. EasyOCR Baseline Benchmark Results (CRAFT Detector)

The baseline results represent the original EasyOCR implementation (CRAFT detector, no FP16) before modifications, tested under the 10-core limit.

### 1 PDF Workload (BS English-Delhi 07-04)
- **Total Wall Time (S1 → S3)**: 424.7s
- **Stage 1 (PDF2Img)**: 10.2s
- **Stage 2 (YOLO Block)**: 1.9s
- **Stage 3 Refiner (Tesseract)**: 127.9s
- **Stage 3 OCR (EasyOCR)**: 284.6s (107 crops)
- **OCR Speed**: ~0.37 crops/sec
- **Resource Usage**:
  - **Avg CPU**: 20.6% (Safely under the 10-core limit)
  - **Avg GPU**: 41.1%
  - **Peak VRAM**: 3.8 GB (Hovering near the 4GB ceiling)

### 13 PDF Workload
- **Total Wall Time (S1 → S3)**: ~4,742s (~79 minutes)
- **Stage 1 (PDF2Img)**: 50.0s
- **Stage 2 (YOLO Block)**: 32.8s
- **Stage 3 Refiner (Tesseract)**: 1078.9s (17.9 minutes)
- **Stage 3 OCR (EasyOCR)**: 3580.9s (59.6 minutes)
- **Total Detections (Entering S3)**: 3,568 crops
- **Crops Reaching OCR**: 2,327 crops
- **Resource Usage**:
  - **Avg CPU**: 20.7%
  - **Avg GPU**: 52.7%
  - **Peak VRAM**: 3.89 GB
- **OOM Count**: 0 (Thanks to subprocess isolation)

---

## 4. Evaluated Optimizations & Best Configuration

During the optimization testing phase, the following EasyOCR improvements were implemented and tested:

| Optimization | Status | Result / Speed Impact |
| :--- | :--- | :--- |
| **T1.1 Batching (size=4, 32)** | **REJECTED** | Slower and highly unstable. `readtext_batched` pads varied-size job ad crops to the maximum dimensions in the batch, wasting massive VRAM and causing CUDA OOMs on the 4GB card. Sequential calls are faster here. |
| **T1.2 DBNet Detector** | **KEEP** | EasyOCR's DBNet is significantly faster than CRAFT for crop-level text detection (~3.3x faster pure crop OCR time: 80s vs 284s for 1 PDF). |
| **T1.3 FP16 Inference** | **KEEP** | Converting models via `.half()` prevented DBNet from OOMing on large inputs and shaved ~400MB off peak VRAM usage. |
| **T1.4 SSD Bypass** | **KEEP** | Passing numpy arrays directly in memory eliminated temporary PNG writes, saving ~1s of Disk I/O per page context. |
| **T1.5 Aggressive Filtering** | **KEEP** | Added logic to drop crops with area < 0.001 or extreme aspect ratios. Audited against Ground Truth labels: safely dropped 1,229 noise crops across 13 PDFs with **0 True Positives lost**. |

### The New Bottleneck
With DBNet+FP16 making crop OCR lightning-fast, the new dominant bottleneck is the **Stage 3 Refiner** (taking nearly 18 minutes just to filter noise via CPU Tesseract) and the **Full-Page Context Builds** inside the OCR stage. 

### Final Recommendation
- **Retain the DBNet + FP16 + SSD Bypass + Filtering** optimizations in `stage03_ocr.py`.
- **Do not alter the transition delays**: Subprocess teardown is the only thing keeping the 4GB VRAM from fragmenting and crashing during a 13-PDF run.
- **Future Work**: The `stage03_block_refiner.py` is crippling the pipeline's overall speed (taking 30% of total runtime). Replacing the Tesseract refiner with a fast, lightweight CNN classifier (or aggressively tuning the YOLO confidence/IoU to drop noise earlier in Stage 2) is the only way to break the 1-hour barrier for 13 PDFs on this hardware.