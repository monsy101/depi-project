#!/usr/bin/env python3
"""
Example: Using the LoRA adapter
"""

from diffusers import StableDiffusionPipeline
import torch
from peft import PeftModel

# Load base model
pipe = StableDiffusionPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)

# Load LoRA adapter
pipe.unet = PeftModel.from_pretrained(pipe.unet, "lora_adapter")

# Generate with LoRA
prompt = "your prompt here, will show learned style"
image = pipe(prompt, num_inference_steps=20).images[0]
image.save("lora_generated.png")
