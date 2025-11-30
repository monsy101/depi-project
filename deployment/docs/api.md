# 🌐 API Documentation

## FastAPI Backend

The deployment includes a high-performance FastAPI backend for Stable Diffusion inference with LoRA support.

### Base URL
```
http://localhost:8000
```

### Authentication
Currently, no authentication is required. For production deployments, consider adding API key authentication.

## Endpoints

### GET /health

Health check endpoint that returns system and model status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "model_status": {
    "loaded": true,
    "lora_available": true,
    "device": "cuda",
    "model_info": {
      "base_model": "stable-diffusion-v1-5",
      "lora_applied": true,
      "torch_version": "2.0.1",
      "cuda_available": true
    }
  }
}
```

### GET /models/status

Returns detailed model loading status.

**Response:**
```json
{
  "loaded": true,
  "lora_available": true,
  "device": "cuda"
}
```

### POST /generate

Generate images from text prompts.

**Request Body:**
```json
{
  "prompt": "abstract geometric patterns in vibrant colors, artistic composition",
  "negative_prompt": "photorealistic, realistic, photographic",
  "num_inference_steps": 20,
  "guidance_scale": 7.5,
  "width": 512,
  "height": 512,
  "num_images": 1,
  "seed": 42
}
```

**Parameters:**
- `prompt` (required): Text description of desired image
- `negative_prompt` (optional): Elements to avoid in generation
- `num_inference_steps` (optional): Number of denoising steps (1-100, default: 20)
- `guidance_scale` (optional): How closely to follow the prompt (1.0-20.0, default: 7.5)
- `width` (optional): Image width in pixels (256-1024, default: 512)
- `height` (optional): Image height in pixels (256-1024, default: 512)
- `num_images` (optional): Number of images to generate (1-4, default: 1)
- `seed` (optional): Random seed for reproducible results

**Response:**
```json
{
  "images": ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."],
  "prompt": "abstract geometric patterns in vibrant colors, artistic composition",
  "negative_prompt": "photorealistic, realistic, photographic",
  "generation_time": 2.34,
  "model_info": {
    "base_model": "stable-diffusion-v1-5",
    "lora_applied": true,
    "device": "cuda",
    "inference_steps": 20,
    "guidance_scale": 7.5,
    "resolution": "512x512"
  }
}
```

### POST /generate/batch

Generate multiple images with different prompts in a single request.

**Request Body:**
```json
[
  {
    "prompt": "abstract geometric patterns",
    "num_inference_steps": 20
  },
  {
    "prompt": "minimalist line art",
    "num_inference_steps": 25
  }
]
```

**Response:**
```json
{
  "results": [
    {
      "index": 0,
      "success": true,
      "result": {
        "images": ["data:image/png;base64,..."],
        "prompt": "abstract geometric patterns",
        "generation_time": 2.34,
        "model_info": {...}
      }
    },
    {
      "index": 1,
      "success": true,
      "result": {
        "images": ["data:image/png;base64,..."],
        "prompt": "minimalist line art",
        "generation_time": 2.67,
        "model_info": {...}
      }
    }
  ]
}
```

### GET /examples

Returns example prompts for testing the API.

**Response:**
```json
{
  "examples": [
    {
      "name": "Abstract Art",
      "prompt": "abstract geometric patterns in vibrant colors, artistic composition",
      "negative_prompt": "photorealistic, realistic, photographic"
    },
    {
      "name": "Minimalist Design",
      "prompt": "minimalist line art, geometric forms, clean design, white background",
      "negative_prompt": "complex, busy, colorful, detailed"
    }
  ]
}
```

### GET /metrics

Returns basic API and system metrics.

**Response:**
```json
{
  "model_loaded": true,
  "lora_available": true,
  "device": "cuda",
  "torch_version": "2.0.1",
  "api_version": "1.0.0"
}
```

## Error Responses

All endpoints return appropriate HTTP status codes and error messages:

- `400 Bad Request`: Invalid request parameters
- `503 Service Unavailable`: Model not loaded or service unavailable
- `500 Internal Server Error`: Unexpected server errors

**Error Response Format:**
```json
{
  "detail": "Error description message"
}
```

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider implementing:
- Request per minute limits
- Concurrent request limits
- API key-based throttling

## Performance Considerations

- **GPU Memory**: Each request consumes ~2-4GB GPU memory
- **Generation Time**: ~2-5 seconds per image depending on parameters
- **Concurrent Requests**: Limited by GPU memory (typically 1-4 concurrent requests)
- **Batch Processing**: More efficient for multiple images with same prompt

## Client Examples

### Python (requests)

```python
import requests
import base64
from PIL import Image
import io

# Generate image
response = requests.post("http://localhost:8000/generate", json={
    "prompt": "abstract geometric patterns in vibrant colors",
    "num_inference_steps": 20,
    "guidance_scale": 7.5
})

if response.status_code == 200:
    data = response.json()
    # Decode and save first image
    image_data = data["images"][0].split(",")[1]  # Remove data:image/png;base64,
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    image.save("generated_image.png")
```

### JavaScript (fetch)

```javascript
async function generateImage() {
    const response = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            prompt: "abstract geometric patterns in vibrant colors",
            num_inference_steps: 20,
            guidance_scale: 7.5
        })
    });

    const data = await response.json();

    // Display image
    const img = document.createElement('img');
    img.src = data.images[0];
    document.body.appendChild(img);
}
```

### cURL

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "abstract geometric patterns in vibrant colors",
    "num_inference_steps": 20,
    "guidance_scale": 7.5
  }'
```

