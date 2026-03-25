import os
from huggingface_hub import hf_hub_download, snapshot_download

# Configuration
CACHE_DIR = os.environ.get("HF_HOME", "./hf_cache")
GGUF_REPO = "Old-Fisherman/SDXL_Finetune_GGUF_Files"
# We pick a specific Q4_K_S file (approx 6-7GB) which is a standard "efficient" quantization
GGUF_FILENAME = "GGUF_Models/juggernautXL_juggXIByRundiffusion_Q4_K_S.gguf"
SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"

print(f"--- Checking Disk Space ---")
# Simple check to ensure we have at least 20GB free before starting
stat = os.statvfs('.')
free_gb = (stat.f_bavail * stat.f_frsize) / 1024**3
print(f"Free space: {free_gb:.2f} GB")
if free_gb < 25:
    print("WARNING: You have less than 25GB free. This might fail.")

print(f"\n--- Downloading GGUF Model ---")
# Downloads just the single .gguf file, not the whole repo
gguf_path = hf_hub_download(
    repo_id=GGUF_REPO,
    filename=GGUF_FILENAME,
    cache_dir=CACHE_DIR
)
print(f"GGUF saved to: {gguf_path}")

print(f"\n--- Downloading SDXL FP16 (Standard) ---")
# Downloads only the fp16 variant to save space
sdxl_path = snapshot_download(
    repo_id=SDXL_REPO,
    allow_patterns=["*.fp16.safetensors", "*.json", "*.txt"],
    ignore_patterns=["*.bin", "*.onnx"],
    cache_dir=CACHE_DIR
)
print(f"SDXL Standard saved to: {sdxl_path}")