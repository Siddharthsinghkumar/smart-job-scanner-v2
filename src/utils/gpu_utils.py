"""GPU resource monitoring and coordination utilities."""

import time
import logging
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def get_gpu_processes(vram_threshold_mb: int = 100) -> List[Dict[str, Any]]:
    """Get list of processes using more than threshold VRAM on any GPU."""
    processes = []
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            # Get compute processes
            apps = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for app in apps:
                vram_mb = app.usedGpuMemory / (1024 * 1024)
                if vram_mb > vram_threshold_mb:
                    try:
                        import psutil
                        proc = psutil.Process(app.pid)
                        name = proc.name()
                    except:
                        name = "unknown"
                    
                    processes.append({
                        "gpu_index": i,
                        "pid": app.pid,
                        "name": name,
                        "used_memory_mb": vram_mb
                    })
        pynvml.nvmlShutdown()
    except Exception as e:
        logger.debug(f"Could not probe GPU processes via pynvml: {e}")
        # Fallback to nvidia-smi if pynvml fails
        try:
            import subprocess
            cmd = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
            output = subprocess.check_output(cmd, encoding="utf-8").strip()
            if output:
                for line in output.split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        pid = int(parts[0])
                        name = parts[1]
                        vram_mb = float(parts[2])
                        if vram_mb > vram_threshold_mb:
                            processes.append({
                                "gpu_index": 0, # Assume index 0 for simplicity in fallback
                                "pid": pid,
                                "name": name,
                                "used_memory_mb": vram_mb
                            })
        except:
            pass

    return processes

def wait_for_gpu(
    vram_threshold_mb: int = 200, 
    check_interval_sec: int = 10, 
    timeout_sec: int = 3600,
    context: str = "OCR/Inference"
):
    """Wait until GPU is clear of other heavy processes."""
    start_time = time.time()
    while True:
        procs = get_gpu_processes(vram_threshold_mb)
        
        # Filter out current process PID
        my_pid = os.getpid()
        other_procs = [p for p in procs if p['pid'] != my_pid]
        
        if not other_procs:
            if time.time() - start_time > 1: # Only log if we actually waited
                logger.info(f"GPU cleared. Proceeding with {context}.")
            return True
            
        proc_names = ", ".join([f"{p['name']}({p['pid']}, {p['used_memory_mb']:.0f}MB)" for p in other_procs])
        logger.info(f"Waiting for GPU to clear. Current processes: {proc_names}. Context: {context}")
        
        if time.time() - start_time > timeout_sec:
            logger.warning(f"Timeout waiting for GPU to clear after {timeout_sec}s. Proceeding anyway...")
            return False
            
        time.sleep(check_interval_sec)
