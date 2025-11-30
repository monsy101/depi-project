# 🚀 Deployment Guide

This guide covers deploying your Stable Diffusion model with LoRA fine-tuning to production environments.

## Prerequisites

- **GPU Requirements**: NVIDIA GPU with CUDA support (minimum 8GB VRAM)
- **System Requirements**:
  - Ubuntu 20.04+ or Windows 10/11 with WSL2
  - Python 3.10+
  - Docker and docker-compose (optional but recommended)
  - 16GB+ RAM
  - 50GB+ free disk space

## Quick Start Deployment

### Option 1: Docker Deployment (Recommended)

1. **Navigate to deployment directory:**
   ```bash
   cd deployment/docker
   ```

2. **Start all services:**
   ```bash
   docker-compose up --build
   ```

3. **Verify deployment:**
   ```bash
   # Check API health
   curl http://localhost:8000/health

   # Check Gradio interface
   # Open http://localhost:7860 in browser
   ```

### Option 2: Direct Python Deployment

1. **Install dependencies:**
   ```bash
   cd deployment
   pip install -r deployment/config/requirements-deployment.txt
   ```

2. **Start FastAPI server:**
   ```bash
   python deployment/api/fastapi_app.py
   ```

3. **Or start Gradio interface:**
   ```bash
   python deployment/api/gradio_app.py
   ```

## Production Deployment Options

### 1. Docker Compose (Recommended for Production)

The included `docker-compose.yml` provides a production-ready setup:

```yaml
version: '3.8'
services:
  stable-diffusion-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Advanced Docker deployment:**
```bash
# Build with no cache
docker-compose build --no-cache

# Run in detached mode
docker-compose up -d

# Scale the service
docker-compose up -d --scale stable-diffusion-api=2

# View logs
docker-compose logs -f stable-diffusion-api
```

### 2. Kubernetes Deployment

For large-scale deployments, use the included Kubernetes manifests:

```bash
# Apply to cluster
kubectl apply -f k8s/

# Check deployment status
kubectl get pods
kubectl get services
```

### 3. Cloud Deployment

#### AWS SageMaker
```bash
# Build and push Docker image
docker build -t stable-diffusion-api .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag stable-diffusion-api:latest <account>.dkr.ecr.us-east-1.amazonaws.com/stable-diffusion-api:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/stable-diffusion-api:latest
```

#### Google Cloud Run
```bash
# Build and deploy
gcloud run deploy stable-diffusion-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --gpu 1 \
  --cpu 4 \
  --memory 16Gi \
  --allow-unauthenticated
```

#### Azure Container Instances
```bash
az container create \
  --resource-group myResourceGroup \
  --name stable-diffusion-api \
  --image <registry>/stable-diffusion-api:latest \
  --cpu 4 \
  --memory 16 \
  --gpu 1 \
  --ports 80 \
  --environment-variables CUDA_VISIBLE_DEVICES=0
```

## Environment Configuration

### Environment Variables

Create a `.env` file in the deployment directory:

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=1

# Model Configuration
MODEL_CACHE_DIR=./models
LORA_ADAPTER_PATH=./models/stable_diffusion_finetune_v1

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/api.log

# Security (for production)
API_KEY=your-secret-api-key-here
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Monitoring
ENABLE_MLFLOW=true
MLFLOW_TRACKING_URI=./mlruns
ENABLE_TENSORBOARD=true
TENSORBOARD_LOG_DIR=./tensorboard_logs
```

### Model Configuration

The deployment automatically detects and loads:
- Base Stable Diffusion model (`models/`)
- LoRA adapter (`models/stable_diffusion_finetune_v1/`)

For custom model paths, modify the environment variables in `docker-compose.yml`.

## Monitoring and Observability

### Health Checks

The API includes comprehensive health monitoring:

```bash
# Health endpoint
curl http://localhost:8000/health

# Model status
curl http://localhost:8000/models/status

# Metrics
curl http://localhost:8000/metrics
```

### Logging

Logs are configured for both development and production:

```bash
# View container logs
docker-compose logs -f stable-diffusion-api

# Structured logging with JSON format for production
# Logs include: timestamp, level, message, request_id, duration
```

### Performance Monitoring

Enable MLflow and TensorBoard for production monitoring:

```bash
# Start MLflow server
mlflow server --backend-store-uri ./mlruns --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000

# Start TensorBoard
tensorboard --logdir ./tensorboard_logs --host 0.0.0.0 --port 6006
```

## Scaling Considerations

### Horizontal Scaling

For high-throughput deployments:

1. **Load Balancer**: Use nginx or AWS ALB
2. **Multiple Instances**: Scale across multiple GPUs
3. **Queue System**: Implement request queuing for peak loads

### GPU Memory Optimization

- **Batch Processing**: Process multiple requests simultaneously
- **Model Quantization**: Use 8-bit quantization for reduced memory usage
- **GPU Memory Pooling**: Implement memory pooling for faster inference

### Caching Strategies

- **Model Warm-up**: Pre-load models on startup
- **Result Caching**: Cache frequent prompts (with TTL)
- **Compiled Models**: Use TorchScript for faster inference

## Security Hardening

### API Security

1. **Enable Authentication:**
   ```python
   # Add to fastapi_app.py
   from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

   security = HTTPBearer()

   @app.post("/generate")
   async def generate_image(
       request: GenerationRequest,
       credentials: HTTPAuthorizationCredentials = Depends(security)
   ):
       # Validate API key
       if credentials.credentials != os.getenv("API_KEY"):
           raise HTTPException(status_code=401, detail="Invalid API key")
   ```

2. **Rate Limiting:**
   ```bash
   # Install slowapi
   pip install slowapi

   # Add to fastapi_app.py
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address
   from slowapi.errors import RateLimitExceeded
   from slowapi.middleware import SlowAPIMiddleware

   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
   app.add_middleware(SlowAPIMiddleware)
   ```

### Container Security

1. **Non-root user:**
   ```dockerfile
   # In Dockerfile
   RUN useradd --create-home --shell /bin/bash app
   USER app
   ```

2. **Minimal base image:**
   ```dockerfile
   FROM nvidia/cuda:12.1-runtime-ubuntu22.04
   ```

3. **Security scanning:**
   ```bash
   # Scan Docker image
   docker scan stable-diffusion-api:latest
   ```

## Troubleshooting

### Common Issues

#### Model Loading Failures
```bash
# Check GPU memory
nvidia-smi

# Verify model files
ls -la models/

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

#### Out of Memory Errors
```bash
# Reduce batch size in API
# Use smaller resolution (256x256 instead of 512x512)
# Enable GPU memory optimization in PyTorch
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### Slow Inference
```python
# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```

### Performance Tuning

```bash
# Profile GPU usage
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv

# Monitor API performance
python -c "
import time
start = time.time()
# Make API request
end = time.time()
print(f'Request time: {end-start:.2f}s')
"
```

## Backup and Recovery

### Model Backup
```bash
# Backup models directory
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Upload to cloud storage
aws s3 cp models_backup_*.tar.gz s3://your-backup-bucket/
```

### Database Backup (if using external DB)
```bash
# Backup MLflow database
sqlite3 mlruns.db .backup mlruns_backup.db

# Or for PostgreSQL
pg_dump mlflow_db > mlflow_backup.sql
```

## Cost Optimization

### GPU Costs
- Use spot instances for development/testing
- Auto-scaling based on demand
- Preemptible instances for batch processing

### Storage Costs
- Compress model artifacts
- Use object storage for backups
- Implement data lifecycle policies

---

## Support

For issues and questions:
1. Check the [troubleshooting guide](./troubleshooting.md)
2. Review [API documentation](./api.md)
3. Check GitHub issues for similar problems
4. Create a new issue with detailed logs and configuration

