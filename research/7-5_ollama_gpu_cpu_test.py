#!/usr/bin/env python3
"""
7-5_ollama_gpu_cpu_test.py
Detect if Ollama is using GPU or CPU by monitoring RAM vs VRAM usage.
"""

import subprocess
import psutil
import requests
import time


def get_vram_usage():
    """Return total VRAM used in MB (NVIDIA only)."""
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        usage = [int(x.strip()) for x in smi.stdout.splitlines() if x.strip().isdigit()]
        return sum(usage)  # total VRAM across GPUs
    except Exception:
        return 0


def get_ram_usage():
    """Return used system RAM in MB."""
    return psutil.virtual_memory().used // (1024 * 1024)


def detect_backend(model="openhermes", timeout=90):
    """
    Detect Ollama backend by comparing RAM vs VRAM usage while loading a model.
    Returns: (backend, delta_vram_MB, delta_ram_MB)
    """
    baseline_vram = get_vram_usage()
    baseline_ram = get_ram_usage()

    print(f"📦 Baseline RAM: {baseline_ram} MB | VRAM: {baseline_vram} MB")
    print(f"🚀 Triggering Ollama model load: {model} ...")

    try:
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": "backend_test", "stream": False},
            timeout=timeout,
        )
    except Exception:
        print("⏳ Model load request sent (may have timed out, that’s fine).")

    # Wait for model to fully load
    time.sleep(20)

    new_vram = get_vram_usage()
    new_ram = get_ram_usage()

    delta_vram = new_vram - baseline_vram
    delta_ram = new_ram - baseline_ram

    backend = "UNKNOWN"
    if delta_vram > 500:  # VRAM spiked
        backend = "GPU"
    elif delta_ram > 500:  # RAM spiked
        backend = "CPU"

    return backend, delta_vram, delta_ram, new_ram, new_vram


if __name__ == "__main__":
    backend, dvram, dram, ram, vram = detect_backend(model="openhermes")

    print("\n📊 Final Test Results:")
    print(f"- Ollama backend detected: {backend}")
    print(f"- RAM change: {dram} MB")
    print(f"- VRAM change: {dvram} MB")
    print(f"- Current RAM: {ram} MB | Current VRAM: {vram} MB")
