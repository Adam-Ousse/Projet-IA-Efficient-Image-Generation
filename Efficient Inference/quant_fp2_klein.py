import torch
import time
import psutil
import os
import gc
import matplotlib.pyplot as plt
import numpy as np
from diffusers import Flux2KleinPipeline
from optimum.quanto import quantize, freeze, qint8
from transformers import CLIPProcessor, CLIPModel

DEVICE = "cuda"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

# 10 Prompts with exactly 5-6 words of target text
PROMPTS = [
    "A neon sign reading 'Welcome to the future city now'",
    "A billboard with the words 'The best coffee in town'",
    "A movie poster clearly titled 'Return of the Space Cowboy'",
    "A handwritten note stating 'Meet me at the station tomorrow'",
    "A storefront awning displaying 'Fresh baked goods every single day'",
    "A protest sign with text 'Save our oceans and marine life'",
    "A vintage typewriter page reading 'It was a dark stormy night'",
    "A graffiti wall tagged with 'Art is the universal human language'",
    "A street sign pointing to 'Historic downtown shopping and dining area'",
    "A book cover boldly titled 'The secret of the lost kingdom'"
]

def get_ram_usage():
    """Returns current RAM usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def calculate_clip_score(images, prompts, clip_model, clip_processor):
    inputs = clip_processor(text=prompts, images=images, return_tensors="pt", padding=True).to(DEVICE)
    outputs = clip_model(**inputs)
    return outputs.logits_per_image.diag().mean().item()

# --- Modifiers ---
def baseline_modifier(pipe):
    print("Keeping baseline FP16...")
    pass

def int8_modifier(pipe):
    print("Quantizing to INT8 on CPU before moving to GPU...")
    quantize(pipe.transformer, weights=qint8)
    freeze(pipe.transformer)

def run_benchmark(run_name, modifier_fn):
    print(f"\n{'='*40}\nStarting Benchmark: {run_name}\n{'='*40}")
    
    # Load to CPU first to prevent VRAM spikes
    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )
    torch.set_grad_enabled(False)

    # Apply Quantization while the model is safely on system RAM
    modifier_fn(pipe)

    # Move to GPU
    pipe = pipe.to(DEVICE)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    metrics = {"vram_gb": [], "ram_gb": [], "time_s": [], "clip_score": []}

    # Warmup run
    _ = pipe(prompt="warmup", num_inference_steps=2).images[0]

    safe_run_name = run_name.replace(" ", "_").lower()

    for i, prompt in enumerate(PROMPTS):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_ram = get_ram_usage()
        start_time = time.time()

        image = pipe(prompt=prompt, num_inference_steps=6).images[0]

        end_time = time.time()
        
        vram_used = torch.cuda.max_memory_allocated() / (1024**3)
        clip_score = calculate_clip_score([image], [prompt], clip_model, clip_processor)

        metrics["vram_gb"].append(vram_used)
        metrics["ram_gb"].append(start_ram) 
        metrics["time_s"].append(end_time - start_time)
        metrics["clip_score"].append(clip_score)
        
        image_filename = f"./results/images/{safe_run_name}_prompt_{i+1:02d}.png"
        image.save(image_filename)
        
        print(f"[{run_name}] Prompt {i+1}/10 done. VRAM: {vram_used:.2f}GB | Time: {end_time-start_time:.2f}s")

    # Aggressive Cleanup
    del pipe
    del clip_model
    del clip_processor
    gc.collect()
    torch.cuda.empty_cache()

    return {k: np.mean(v) for k, v in metrics.items()}

def plot_results(results_dict):
    labels = list(results_dict.keys())
    metrics = ["vram_gb", "ram_gb", "time_s", "clip_score"]
    titles = ["Peak VRAM (GB) ↓", "RAM Usage (GB) ↓", "Inference Time (s) ↓", "Image Quality (CLIP Score) ↑"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('FLUX.2-Klein 4B Optimization Benchmark', fontsize=16, fontweight='bold')
    
    # Colors for FP16 and INT8
    colors = ['#4C72B0', '#55A868']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        values = [results_dict[label][metric] for label in labels]
        
        # Adjust bar width since we only have two bars now
        bars = ax.bar(labels, values, color=colors, width=0.5)
        ax.set_title(title, fontsize=14)
        ax.set_ylabel(metric.split('_')[0].upper())
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.02), f"{yval:.2f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("./results/flux_4b_benchmark.png", dpi=300)
    print("\nGraphs saved to ./results/flux_4b_benchmark.png")

def main():
    os.makedirs("./results/images", exist_ok=True)
    all_results = {}
    
    # 1. 4B Baseline
    all_results["4B FP16 Baseline"] = run_benchmark("4B FP16 Baseline", baseline_modifier)
    
    # 2. 4B Quantized
    all_results["4B INT8 Quantized"] = run_benchmark("4B INT8 Quantized", int8_modifier)
    
    plot_results(all_results)
    print("\nAll benchmarking complete. Check the ./results folder for images and graphs.")

if __name__ == "__main__":
    main()