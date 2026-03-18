import torch
try:
    torch.cuda.init()
    print("CUDA ready:", torch.cuda.get_device_name(0))
except Exception as e:
    print("⚠️ CUDA init failed:", e)
    # fall back to CPU or exit cleanly
