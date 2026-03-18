import time
import torch
import os
import psutil
import platform
import argostranslate.package
import argostranslate.translate
import subprocess

# ----------------- Argos Translate Model Setup -----------------
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
hi_en_package = next(
    (p for p in available_packages if p.from_code == "hi" and p.to_code == "en"), None
)
if hi_en_package:
    argostranslate.package.install_from_path(hi_en_package.download())

# ----------------- Translation Function -----------------
def translate_hi_en(text: str) -> str:
    start_time = time.time()
    translated = argostranslate.translate.translate(text, "hi", "en")
    end_time = time.time()
    translation_time = end_time - start_time
    return translated, translation_time

# ----------------- Test -----------------
hindi_text = "नौकरी के लिए आवेदन आमंत्रित हैं।"
print("📝 Input:", hindi_text)

english_text, translation_time = translate_hi_en(hindi_text)
print("➡️ Translation:", english_text)

# ----------------- Summary -----------------
print("\n📊 SUMMARY")
print("Translation Time: {:.2f}s".format(translation_time))
print("🔎 Torch version:", torch.__version__)
print("🔎 CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("🔎 Device: GPU")
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)
    try:
        driver_version = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ).decode().strip()
        print("Driver Version:", driver_version)
    except Exception as e:
        print("Driver Version: Could not detect:", e)
else:
    print("🔎 Device: CPU")
    cpu_name = platform.processor()
    total_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    print(f"CPU Name: {cpu_name}")
    print(f"Total Physical Cores: {total_cores}")
    print(f"Logical Cores: {logical_cores}")
