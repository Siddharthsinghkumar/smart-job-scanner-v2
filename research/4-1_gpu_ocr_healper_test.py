import easyocr
import torch
reader = easyocr.Reader(['hi', 'en'], gpu=True)
print("Models have been downloaded.")
print("Torch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Torch device:", torch.cuda.get_device_name(0))

reader = easyocr.Reader(['en'], gpu=True)
print("✅ EasyOCR initialized with GPU")