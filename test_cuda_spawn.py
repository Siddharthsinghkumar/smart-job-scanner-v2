import os
import sys

def worker():
    import torch
    print(f"Child PID: {os.getpid()}")
    try:
        if not torch.cuda.is_available():
            print("CUDA NOT AVAILABLE in child.")
        else:
            print(f"CUDA Ready in child: {torch.cuda.get_device_name(0)}")
            torch.cuda.init()
            print("CUDA Init Success in child.")
    except Exception as e:
        print(f"Error in child: {e}")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    print(f"Parent PID: {os.getpid()}")
    mp.set_start_method('spawn', force=True)
    p = mp.Process(target=worker)
    p.start()
    p.join()
