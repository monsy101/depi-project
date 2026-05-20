"""
Lightweight FastAPI app for Vercel serverless.

Stable Diffusion inference needs a GPU and cannot run in Vercel's 500MB limit.
Deploy the full API with Docker (see deployment/docs/deployment.md), or set
INFERENCE_API_URL to proxy /generate to that backend.
"""

import os
from datetime import datetime
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

INFERENCE_API_URL = os.getenv("INFERENCE_API_URL", "").rstrip("/")

app = FastAPI(
    title="Stable Diffusion API",
    description="Vercel gateway for the depi-project API. Inference runs on a GPU backend.",
    version="1.0.0",
)


class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field("", description="Negative prompt")
    num_inference_steps: Optional[int] = Field(20, ge=1, le=100)
    guidance_scale: Optional[float] = Field(7.5, ge=1.0, le=20.0)
    width: Optional[int] = Field(512, ge=256, le=1024)
    height: Optional[int] = Field(512, ge=256, le=1024)
    num_images: Optional[int] = Field(1, ge=1, le=4)
    seed: Optional[int] = Field(None)


class ModelStatus(BaseModel):
    loaded: bool
    lora_available: bool
    device: str
    model_info: Optional[dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    platform: str
    inference_backend: Optional[str]
    model_status: ModelStatus


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": "depi-project",
        "platform": "vercel",
        "inference": INFERENCE_API_URL or None,
        "message": (
            "This deployment is an API gateway. "
            "Run inference via Docker/GPU or set INFERENCE_API_URL."
        ),
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    backend_ready = False
    if INFERENCE_API_URL:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{INFERENCE_API_URL}/health")
                backend_ready = response.status_code == 200
        except httpx.HTTPError:
            backend_ready = False

    return HealthResponse(
        status="healthy" if backend_ready or not INFERENCE_API_URL else "degraded",
        timestamp=datetime.now().isoformat(),
        platform="vercel",
        inference_backend=INFERENCE_API_URL or None,
        model_status=ModelStatus(
            loaded=backend_ready,
            lora_available=False,
            device="remote" if INFERENCE_API_URL else "not-configured",
            model_info={
                "note": "Models run on the GPU backend, not on Vercel.",
                "configure": "Set INFERENCE_API_URL to your Docker/GPU API base URL.",
            },
        ),
    )


@app.get("/models/status", response_model=ModelStatus)
async def get_model_status() -> ModelStatus:
    if not INFERENCE_API_URL:
        return ModelStatus(
            loaded=False,
            lora_available=False,
            device="not-configured",
            model_info={"message": "Set INFERENCE_API_URL to a running GPU API."},
        )
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{INFERENCE_API_URL}/models/status")
        response.raise_for_status()
        return ModelStatus(**response.json())


@app.post("/generate")
async def generate_image(request: GenerationRequest) -> Any:
    if not INFERENCE_API_URL:
        raise HTTPException(
            status_code=503,
            detail=(
                "Image generation is not available on Vercel. "
                "Deploy deployment/deployment/api/fastapi_app.py with Docker on a GPU host, "
                "then set INFERENCE_API_URL to that server's base URL."
            ),
        )
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{INFERENCE_API_URL}/generate",
            json=request.model_dump(exclude_none=True),
        )
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()
