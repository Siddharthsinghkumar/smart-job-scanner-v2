import torch
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)
if torch.cuda.is_available():
    x = torch.rand(2,2).cuda()
    print("Tensor on:", x.device)

print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")    