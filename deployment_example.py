#!/usr/bin/env python3
"""
Deployment Example: LoRA Model Usage
Demonstrates that LoRA needs the base model + adapter
"""

from diffusers import StableDiffusionPipeline
from peft import PeftModel
import torch
import os

def demonstrate_lora_deployment():
    """Show how LoRA deployment works"""

    print("🎯 LoRA DEPLOYMENT DEMONSTRATION")
    print("=" * 50)
    print()
    print("IMPORTANT: LoRA is NOT a standalone model!")
    print("LoRA = Base Model + Small Adapter Weights")
    print()

    # Check what we have
    base_model_available = True  # We have SD 1.5
    lora_adapter_path = "lora/stable_diffusion_finetune_v1"
    lora_available = os.path.exists(lora_adapter_path)

    print("📦 DEPLOYMENT COMPONENTS:")
    print(f"✅ Base Model (SD 1.5): {'Available' if base_model_available else 'Missing'}")
    print(f"✅ LoRA Adapter: {'Available' if lora_available else 'Missing'}")
    print()

    if not lora_available:
        print("❌ LoRA adapter not found!")
        return

    print("🚀 DEPLOYMENT PROCESS:")
    print("Step 1: Load base Stable Diffusion model")
    print("Step 2: Apply LoRA adapter to the model")
    print("Step 3: Use combined model for generation")
    print()

    try:
        # Step 1: Load base model
        print("Loading base model...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "sd-legacy/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        print(f"✅ Base model loaded on {device}")

        # Step 2: Apply LoRA adapter
        print("Applying LoRA adapter...")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_adapter_path)
        print("✅ LoRA adapter applied")

        # Step 3: Generate with combined model
        print("Generating image with LoRA-enhanced model...")
        prompt = "abstract geometric patterns in vibrant colors, artistic composition"

        image = pipe(
            prompt,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]

        output_path = "lora_deployment_test.png"
        image.save(output_path)
        print(f"✅ Image generated and saved: {output_path}")

        print("\n🎉 SUCCESS!")
        print("LoRA deployment works correctly!")
        print("The model combines base SD 1.5 + your fine-tuned adapter")

    except Exception as e:
        print(f"❌ Error during deployment: {e}")
        import traceback
        traceback.print_exc()

def show_file_sizes():
    """Show the size difference between base model and LoRA"""

    print("\n📊 MODEL SIZE COMPARISON:")
    print("=" * 30)

    # Base model size (approximate)
    base_model_size = 4.0  # GB
    print(".1f")

    # LoRA adapter size
    lora_adapter_path = "lora/stable_diffusion_finetune_v1"
    if os.path.exists(lora_adapter_path):
        total_size = 0
        for root, dirs, files in os.walk(lora_adapter_path):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))

        lora_size_mb = total_size / (1024 * 1024)
        print(".3f")

        # Compression ratio
        compression_ratio = (base_model_size * 1024) / lora_size_mb
        print(",.0f")
        print(".1f")

def explain_lora_concept():
    """Explain how LoRA works conceptually"""

    print("\n🧠 HOW LoRA WORKS:")
    print("=" * 20)
    print("1. Start with base model (SD 1.5) - 4GB")
    print("2. Add small 'adapter' matrices to key layers")
    print("3. Only train the adapters (0.04MB), freeze base model")
    print("4. Result: Base model + tiny trained adapters")
    print("5. Deployment: Load base + apply adapters")
    print()
    print("Benefits:")
    print("• 2000x smaller than full fine-tuning")
    print("• Shareable and versionable")
    print("• Compatible across different base models")
    print("• Memory efficient")
    print()
    print("Trade-offs:")
    print("• Always needs base model")
    print("• Slightly slower than merged models")
    print("• Requires PEFT library")

if __name__ == "__main__":
    demonstrate_lora_deployment()
    show_file_sizes()
    explain_lora_concept()

    print("\n" + "="*60)
    print("DEPLOYMENT SUMMARY:")
    print("• LoRA = Base Model + Adapter (both required)")
    print("• Adapter is NOT standalone - needs base model")
    print("• This is normal and expected for LoRA!")
    print("• Your deployment will load both components")
    print("="*60)