#!/usr/bin/env python3
"""
Test script for Stable Diffusion v1.5 model
Generates a sample image to verify the model is working
"""

import torch
import os
from diffusers import StableDiffusionPipeline
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_local_model():
    """
    Load the locally downloaded Stable Diffusion model components individually
    """
    print("[LOAD] Loading Stable Diffusion model components...")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[HARDWARE] Using device: {device.upper()}")

        # Load components individually
        from transformers import CLIPTextModel, CLIPTokenizer
        from diffusers import UNet2DConditionModel, AutoencoderKL

        print("[LOAD] Loading tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained("models/tokenizer")

        print("[LOAD] Loading text encoder...")
        text_encoder = CLIPTextModel.from_pretrained("models/text_encoder")
        text_encoder = text_encoder.to(device)

        print("[LOAD] Loading UNet...")
        unet = UNet2DConditionModel.from_pretrained("models/unet")
        unet = unet.to(device)

        print("[LOAD] Loading VAE...")
        vae = AutoencoderKL.from_pretrained("models/vae")
        vae = vae.to(device)

        print("[SUCCESS] All model components loaded successfully!")
        return {
            'tokenizer': tokenizer,
            'text_encoder': text_encoder,
            'unet': unet,
            'vae': vae
        }, device

    except Exception as e:
        print(f"[ERROR] Failed to load model components: {str(e)}")
        return None, None

def generate_test_image(components, device, prompt, output_path="test_output.png"):
    """
    Generate a test image using individual model components
    """
    print(f"[GENERATE] Creating image for prompt: '{prompt}'")
    print("[TIME] This may take 30-60 seconds on CPU...")

    try:
        tokenizer = components['tokenizer']
        text_encoder = components['text_encoder']
        unet = components['unet']
        vae = components['vae']

        # Tokenize prompt
        print("[PROCESS] Tokenizing prompt...")
        text_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        # Encode text
        print("[PROCESS] Encoding text...")
        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

        # Create unconditional embeddings for classifier-free guidance
        uncond_input = tokenizer(
            "",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

        # Concatenate embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Generate latent noise
        print("[PROCESS] Generating latent space...")
        latents = torch.randn((1, 4, 64, 64)).to(device)

        # Denoising loop (simplified diffusion)
        print("[PROCESS] Running diffusion process...")
        from diffusers import DDPMScheduler
        scheduler = DDPMScheduler.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="scheduler")

        scheduler.set_timesteps(20)  # Fewer steps for faster testing

        with torch.no_grad():
            for t in scheduler.timesteps:
                # Expand latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # Predict noise
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

                # Update latents
                latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents to image
        print("[PROCESS] Decoding to image...")
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            image = vae.decode(latents).sample

        # Convert to PIL image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype("uint8")
        image = Image.fromarray(image)

        # Save the image
        image.save(output_path)
        print(f"[SUCCESS] Image saved to: {output_path}")

        # Display basic info
        print(f"[INFO] Image size: {image.size}")
        print(f"[INFO] Image mode: {image.mode}")

        return image

    except Exception as e:
        print(f"[ERROR] Failed to generate image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("[TARGET] STABLE DIFFUSION V1.5 MODEL TEST")
    print("=" * 50)

    # Check if model exists
    if not os.path.exists("models"):
        print("[ERROR] Model directory not found! Please run prepare_model.py first.")
        return

    # Load model
    components, device = load_local_model()
    if components is None:
        return

    print(f"[HARDWARE] Running on: {device.upper()}")
    if device == "cpu":
        print("[WARNING] CPU mode - generation will be slower")

    print("\n" + "=" * 50)

    # Single test prompt first
    test_prompt = "a beautiful sunset over mountains, photorealistic"
    print(f"\n[TEST] Generating image...")
    output_path = "test_image.png"

    image = generate_test_image(components, device, test_prompt, output_path)

    if image is not None:
        print("[SUCCESS] Model test completed successfully!")
    else:
        print("[FAILED] Model test failed!")
        return

    print("\n" + "=" * 50)
    print("[COMPLETE] Model testing finished!")
    print("[FILES] Check the generated image:")
    print("  - test_image.png")

    print("\n[TIPS] If the image looks good, your model is ready for fine-tuning!")
    print("       If not, check the error messages above.")

if __name__ == "__main__":
    main()
