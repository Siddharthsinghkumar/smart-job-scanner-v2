import os
import torch

# Force single GPU visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    print("Attempting torch.cuda.init()...")
    torch.cuda.init()
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    t = torch.cuda.FloatTensor([1.0])
    print("Successfully allocated tensor on GPU.")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
