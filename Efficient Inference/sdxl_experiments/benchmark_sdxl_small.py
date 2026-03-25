import torch
import time
import gc
import os
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

# --- CONFIGURATION ---
PROMPT = "A delicate  square cake, cream and fruit, with 'CHEERS to the GRADUATE' written on it"
# PROMPT = "A futuristic billboard in Tokyo displaying the text 'GENERATION' in bright neon blue letters, high detail, 8k"
OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def benchmark_standard():
    print(f"\n=== [1/2] Running Standard SDXL (FP16) ===")
    
    # Standard Load
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    
    pipe(prompt="warmup", num_inference_steps=1)
    
    print(f"Generating Standard: '{PROMPT}'")
    start = time.perf_counter()
    image = pipe(prompt=PROMPT, num_inference_steps=30).images[0]
    end = time.perf_counter()
    
    print(f"Standard Time: {end - start:.4f}s")
    image.save(f"{OUTPUT_DIR}/standard_sdxl.png")
    
    # Cleanup
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    return end - start

if __name__ == "__main__":
    t_std = benchmark_standard()
    
    print("\n" + "="*30)
    print("FINAL RESULTS")
    print(f"Standard (FP16):   {t_std:.4f}s")