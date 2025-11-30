#!/usr/bin/env python3
"""
Gradio web interface for Stable Diffusion with LoRA support
"""

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import os

# Global variables for models
pipe = None
lora_available = False

def load_models():
    """Load Stable Diffusion model and LoRA adapter"""
    global pipe, lora_available

    if pipe is None:
        print("Loading Stable Diffusion model...")

        # Load base model
        pipe = StableDiffusionPipeline.from_pretrained(
            "sd-legacy/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        print(f"Model loaded on {device}")

        # Check for LoRA adapter
        lora_path = "lora_adapter"
        if os.path.exists(lora_path) and os.path.exists(os.path.join(lora_path, "pytorch_lora_weights.bin")):
            try:
                print("Loading LoRA adapter...")
                pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
                lora_available = True
                print("LoRA adapter loaded successfully!")
            except Exception as e:
                print(f"LoRA loading failed: {e}")
                lora_available = False
        else:
            print("No LoRA adapter found")

    return pipe is not None

def generate_image(prompt, negative_prompt="", num_steps=20, guidance_scale=7.5, width=512, height=512):
    """Generate image from prompt"""

    if pipe is None:
        return None, "Model not loaded. Please try again."

    try:
        # Generate image
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=int(num_steps),
                guidance_scale=float(guidance_scale),
                width=int(width),
                height=int(height)
            ).images[0]

        status = "Image generated successfully!"
        if lora_available:
            status += " (with LoRA adapter applied)"

        return image, status

    except Exception as e:
        return None, f"Generation failed: {str(e)}"

def create_interface():
    """Create Gradio interface"""

    # Load models on startup
    load_models()

    # Create interface
    with gr.Blocks(title="Stable Diffusion with LoRA", theme=gr.themes.Soft()) as interface:

        gr.Markdown("""
        # 🎨 Stable Diffusion with LoRA Fine-tuning

        Generate images using your fine-tuned Stable Diffusion model with LoRA adapter support.
        """)

        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3,
                    value="abstract geometric patterns in vibrant colors, artistic composition"
                )

                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt (optional)",
                    placeholder="What to avoid in the image...",
                    lines=2
                )

                with gr.Row():
                    steps_slider = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=20,
                        step=5,
                        label="Inference Steps"
                    )

                    guidance_slider = gr.Slider(
                        minimum=1.0,
                        maximum=15.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale"
                    )

                with gr.Row():
                    width_slider = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=64,
                        label="Width"
                    )

                    height_slider = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=64,
                        label="Height"
                    )

                generate_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")

            with gr.Column():
                output_image = gr.Image(label="Generated Image", height=512)
                status_text = gr.Textbox(label="Status", interactive=False)

        # LoRA status indicator
        lora_status = "✅ LoRA adapter loaded" if lora_available else "❌ No LoRA adapter found"
        gr.Markdown(f"**Model Status:** {lora_status}")

        # Connect the interface
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt_input, negative_prompt_input, steps_slider, guidance_slider, width_slider, height_slider],
            outputs=[output_image, status_text]
        )

        # Examples
        gr.Examples(
            examples=[
                ["abstract geometric patterns in vibrant colors, artistic composition", "", 20, 7.5, 512, 512],
                ["colorful gradient background with flowing shapes, modern art", "blurry, low quality", 25, 8.0, 512, 512],
                ["minimalist line art, geometric forms, clean design", "complex, busy", 30, 9.0, 512, 512],
            ],
            inputs=[prompt_input, negative_prompt_input, steps_slider, guidance_slider, width_slider, height_slider],
            label="Example Prompts"
        )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
