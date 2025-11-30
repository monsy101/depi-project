#!/usr/bin/env python3
"""
Stable Diffusion v1.5 Model Preparation Script
Downloads and sets up the base model for fine-tuning
"""

import os
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_stable_diffusion_model():
    """
    Download Stable Diffusion v1.5 model from Hugging Face
    """
    logger.info("[START] Starting Stable Diffusion v1.5 model download...")
    print("[DOWNLOAD] Downloading Stable Diffusion v1.5 model...")
    print("[TIME] This may take 5-10 minutes depending on your internet speed")
    print("[PROGRESS] Progress will be shown below:")

    model_id = "sd-legacy/stable-diffusion-v1-5"

    try:
        # Download the complete pipeline with progress indication
        logger.info(f"[CONNECT] Connecting to Hugging Face: {model_id}")
        print(f"[MODEL] Model ID: {model_id}")

        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # Use float32 for CPU
            safety_checker=None,  # Disable safety checker for training
            requires_safety_checker=False,
            # Add progress callback
        )

        print("[SUCCESS] Model downloaded successfully!")

        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)

        # Save the model components
        logger.info("Saving model components...")

        # Save tokenizer
        pipe.tokenizer.save_pretrained("models/tokenizer")

        # Save text encoder
        pipe.text_encoder.save_pretrained("models/text_encoder")

        # Save UNet
        pipe.unet.save_pretrained("models/unet")

        # Save VAE
        pipe.vae.save_pretrained("models/vae")

        logger.info("✅ Model download and save completed successfully!")
        logger.info("Model components saved to 'models/' directory")

        # Print model info
        print("\nModel Information:")
        print(f"- Tokenizer: {pipe.tokenizer.__class__.__name__}")
        print(f"- Text Encoder: {pipe.text_encoder.__class__.__name__}")
        print(f"- UNet: {pipe.unet.__class__.__name__}")
        print(f"- VAE: {pipe.vae.__class__.__name__}")

        # Test basic functionality
        logger.info("[TEST] Testing model with a simple prompt...")
        test_prompt = "a photograph of a cat"
        with torch.no_grad():
            # Tokenize
            tokens = pipe.tokenizer(test_prompt, return_tensors="pt")
            print(f"[TEST] Tokenization successful: {tokens['input_ids'].shape}")

        return True

    except Exception as e:
        logger.error(f"[ERROR] Error downloading model: {str(e)}")
        return False

def main():
    print("[TARGET] STABLE DIFFUSION V1.5 MODEL PREPARATION")
    print("=" * 60)

    # Check hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[HARDWARE] Device: {device.upper()}")
    if device == "cpu":
        print("[WARNING] Using CPU - training will be slower but functional")
    else:
        print("[GPU] CUDA detected - training will be much faster!")

    print("\n" + "=" * 60)

    # Download model
    success = download_stable_diffusion_model()

    print("\n" + "=" * 60)
    if success:
        print("[SUCCESS] MODEL PREPARATION COMPLETED!")
        print("\n[STEPS] Next steps:")
        print("[DONE] 1. Wait for your dataset to finish downloading")
        print("[WAIT] 2. Run training script once dataset is ready")
        print("[CONFIG] 3. Monitor training progress and adjust parameters as needed")
    else:
        print("[ERROR] MODEL PREPARATION FAILED")
        print("[DEBUG] Please check the error messages above and try again.")

if __name__ == "__main__":
    main()
