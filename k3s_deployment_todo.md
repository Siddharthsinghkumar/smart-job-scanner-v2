# Local Deployment TODO: TrueNAS k3s & Docker

This document outlines the deferred steps required to deploy `smart-job-scanner-v2` locally to a TrueNAS server using its built-in k3s (Kubernetes) service.

## 1. Dockerization
- [ ] **Create `Dockerfile`**:
  - Base image: Use a lightweight NVIDIA CUDA base image (e.g., `nvidia/cuda:12.1.0-runtime-ubuntu22.04`) to support PyTorch, EasyOCR, and SentenceTransformers on the GPU.
  - Install system dependencies: `tesseract-ocr`, `tesseract-ocr-deu` (and other languages), `libgl1` (for OpenCV), `poppler-utils` (for PDF processing).
  - Install Python 3.12 and create a virtual environment.
  - Copy `requirements.txt` and install Python dependencies.
  - Copy the `src/`, `data/` (as mount points), and `tests/` directories.
- [ ] **Create `.dockerignore`**: Exclude local `logs/`, `venv/`, `4_env/`, `__pycache__/`, and large local models to keep the image size small.
- [ ] **Local Testing (`docker-compose.yml`)**: Create a compose file to test the containerized pipeline locally before pushing it to TrueNAS. Ensure volume mounts map correctly for the `data/` directory.

## 2. Kubernetes (k3s) Manifests
- [ ] **Namespace**: Create a dedicated namespace (e.g., `smart-job-scanner`).
- [ ] **Persistent Volumes (PV & PVC)**:
  - Define a `PersistentVolumeClaim` that maps to a TrueNAS dataset (SMB/NFS share or HostPath) so that `data/` (resumes, extracted PDFs, shortlisted jobs) and `logs/` (structlog JSONL files) are persisted across container restarts.
- [ ] **Deployment / CronJob**:
  - If the pipeline runs continuously, create a `Deployment` with 1 replica.
  - If the pipeline runs on a schedule (e.g., every 6 hours), create a `CronJob` manifest.
  - Define resource requests/limits, specifically requesting `nvidia.com/gpu: 1` to ensure the TrueNAS GPU is allocated to the pod.
- [ ] **ConfigMap & Secrets**:
  - Store environment variables (e.g., `MERLIN_API_KEY`, `TELEGRAM_TOKEN`, etc.) securely using k8s `Secret`.
  - Store non-sensitive configuration in a `ConfigMap`.

## 3. Observability & Metrics (Prometheus)
- [ ] **Service**: Expose port `8765` (our Prometheus metrics port) internally within the k3s cluster.
- [ ] **ServiceMonitor / PodMonitor**: Add a `ServiceMonitor` manifest to instruct the TrueNAS Prometheus instance to scrape metrics from our pod on port `8765`.
- [ ] **Grafana Dashboard**: Create a Grafana dashboard JSON model to visualize `jobv2_jobs_processed_total`, `jobv2_stage_duration_seconds`, and `jobv2_pipeline_running`.

## 4. TrueNAS Specific Setup
- [ ] **Registry**: Decide whether to push the Docker image to Docker Hub, a private registry, or build it directly on the TrueNAS host.
- [ ] **App Deployment**: Apply the manifests via `kubectl` on the TrueNAS scale shell, or package them into a TrueNAS Custom App (Helm chart) via the TrueNAS UI.

---
**Status**: DEFERRED  
**Next Steps**: When ready to deploy, work through these checkboxes sequentially.
