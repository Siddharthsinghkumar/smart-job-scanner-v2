import os
import sys
import torch
import multiprocessing as mp

def worker():
    print(f"Child PID: {os.getpid()}")
    try:
        if not torch.cuda.is_available():
            print("CUDA NOT AVAILABLE in child.")
        else:
            print(f"CUDA Ready in child: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Error in child: {e}")

if __name__ == "__main__":
    print(f"Parent PID: {os.getpid()}")
    # Use default fork
    p = mp.Process(target=worker)
    p.start()
    p.join()
