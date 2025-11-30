#!/usr/bin/env python3
"""
FastAPI Backend for Stable Diffusion Model Deployment
Provides REST API endpoints for model inference and management
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import uvicorn
import os
import logging
import time
from datetime import datetime
import base64
from io import BytesIO
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stable Diffusion API",
    description="REST API for Stable Diffusion model inference with LoRA support",
    version="1.0.0"
)

# Global variables for models
model_loaded = False
pipe = None
lora_available = False

# Request/Response Models
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field("", description="Negative prompt to avoid certain elements")
    num_inference_steps: Optional[int] = Field(20, ge=1, le=100, description="Number of inference steps")
    guidance_scale: Optional[float] = Field(7.5, ge=1.0, le=20.0, description="Guidance scale for prompt adherence")
    width: Optional[int] = Field(512, ge=256, le=1024, description="Image width")
    height: Optional[int] = Field(512, ge=256, le=1024, description="Image height")
    num_images: Optional[int] = Field(1, ge=1, le=4, description="Number of images to generate")
    seed: Optional[int] = Field(None, description="Random seed for reproducible generation")

class GenerationResponse(BaseModel):
    images: List[str]  # Base64 encoded images
    prompt: str
    negative_prompt: str
    generation_time: float
    model_info: Dict[str, Any]

class ModelStatus(BaseModel):
    loaded: bool
    lora_available: bool
    device: str
    model_info: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_status: ModelStatus

def load_models():
    """Load Stable Diffusion model and LoRA adapter"""
    global pipe, model_loaded, lora_available

    if model_loaded:
        return

    try:
        logger.info("Loading Stable Diffusion model...")

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
        logger.info(f"Model loaded on {device}")

        # Check for LoRA adapter
        lora_path = "models/stable_diffusion_finetune_v1"
        if os.path.exists(lora_path) and os.path.exists(os.path.join(lora_path, "pytorch_lora_weights.bin")):
            try:
                logger.info("Loading LoRA adapter...")
                pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
                lora_available = True
                logger.info("LoRA adapter loaded successfully!")
            except Exception as e:
                logger.warning(f"LoRA loading failed: {e}")
                lora_available = False
        else:
            logger.info("No LoRA adapter found")

        model_loaded = True

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_status=ModelStatus(
            loaded=model_loaded,
            lora_available=lora_available,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_info={
                "base_model": "stable-diffusion-v1-5",
                "lora_applied": lora_available,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available()
            }
        )
    )

@app.get("/models/status", response_model=ModelStatus)
async def get_model_status():
    """Get current model status"""
    return ModelStatus(
        loaded=model_loaded,
        lora_available=lora_available,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    """Generate images from text prompt"""

    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not pipe:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        start_time = time.time()

        # Set seed if provided
        if request.seed is not None:
            torch.manual_seed(request.seed)

        # Generate images
        with torch.no_grad():
            output = pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                width=request.width,
                height=request.height,
                num_images_per_prompt=request.num_images
            )

        generation_time = time.time() - start_time

        # Convert images to base64
        image_data = []
        for image in output.images:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_data.append(f"data:image/png;base64,{img_str}")

        # Prepare response
        response = GenerationResponse(
            images=image_data,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or "",
            generation_time=generation_time,
            model_info={
                "base_model": "stable-diffusion-v1-5",
                "lora_applied": lora_available,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "resolution": f"{request.width}x{request.height}"
            }
        )

        logger.info(".2f")

        return response

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate/batch")
async def generate_batch(requests: List[GenerationRequest]):
    """Generate multiple images in batch"""
    results = []

    for i, request in enumerate(requests):
        try:
            result = await generate_image(request)
            results.append({
                "index": i,
                "success": True,
                "result": result
            })
        except Exception as e:
            results.append({
                "index": i,
                "success": False,
                "error": str(e)
            })

    return {"results": results}

@app.get("/examples")
async def get_examples():
    """Get example prompts for testing"""
    examples = [
        {
            "name": "Abstract Art",
            "prompt": "abstract geometric patterns in vibrant colors, artistic composition",
            "negative_prompt": "photorealistic, realistic, photographic"
        },
        {
            "name": "Minimalist Design",
            "prompt": "minimalist line art, geometric forms, clean design, white background",
            "negative_prompt": "complex, busy, colorful, detailed"
        },
        {
            "name": "Gradient Background",
            "prompt": "colorful gradient background with flowing shapes, modern art",
            "negative_prompt": "sharp edges, geometric, structured"
        }
    ]

    return {"examples": examples}

@app.get("/metrics")
async def get_metrics():
    """Get basic API metrics"""
    # In a production system, you'd track more detailed metrics
    return {
        "model_loaded": model_loaded,
        "lora_available": lora_available,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "api_version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1  # Set to 1 for GPU memory management
    )

