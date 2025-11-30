# 🚀 Stable Diffusion Backend Service

Quick guide to run your Stable Diffusion backend API using Docker.

## Prerequisites

- **Docker Desktop** installed and running
- **NVIDIA GPU** (optional but recommended for faster inference)
- **Windows/Linux/macOS** with Docker support

## Quick Start

### Step 1: Start Docker Desktop
Make sure Docker Desktop is running on your system.

### Step 2: Run the Backend Service

**Windows:**
```batch
# From the deployment directory
start_backend.bat
```

**Linux/macOS:**
```bash
# From the deployment directory
./start_backend.sh
```

**Manual Docker Commands:**
```bash
# Navigate to docker directory
cd deployment/deployment/docker

# Build and start services
docker-compose up --build -d

# Check status
docker-compose ps
```

## 🌐 Available Services

Once started, your services will be available at:

- **FastAPI Backend API**: http://localhost:8000
  - Health check: `GET /health`
  - Generate images: `POST /generate`
  - API docs: http://localhost:8000/docs

- **Gradio Web Interface**: http://localhost:7860
  - User-friendly web UI for testing

## 🧪 Test the API

### Health Check
```bash
curl http://localhost:8000/health
```

### Generate an Image
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "abstract geometric patterns in vibrant colors",
    "num_inference_steps": 20,
    "guidance_scale": 7.5
  }'
```

### Python Test
```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "prompt": "abstract geometric patterns in vibrant colors",
    "num_inference_steps": 20,
    "guidance_scale": 7.5
})

if response.status_code == 200:
    data = response.json()
    print("Generated image!")
    # data['images'][0] contains base64 encoded image
```

## 📊 Monitoring

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f fastapi-backend
```

### Service Status
```bash
docker-compose ps
```

### Resource Usage
```bash
# Check GPU usage
nvidia-smi

# Check Docker resource usage
docker stats
```

## 🛑 Stop Services

```bash
# From docker directory
docker-compose down

# Or from deployment directory
cd deployment/deployment/docker && docker-compose down
```

## 🔧 Troubleshooting

### Docker Not Running
- Start Docker Desktop
- Wait for Docker daemon to start
- Try again

### Port Already in Use
- Check what's using ports 8000/7860: `netstat -ano | findstr :8000`
- Change ports in `docker-compose.yml` if needed

### GPU Not Available
- Ensure NVIDIA drivers are installed
- Check GPU status: `nvidia-smi`
- Services will fall back to CPU (slower)

### Model Loading Issues
- Check model files exist: `ls deployment/models/`
- Check Docker volume mounts
- View logs: `docker-compose logs fastapi-backend`

### Build Failures
- Clear Docker cache: `docker system prune -a`
- Rebuild: `docker-compose build --no-cache`

## 📁 Project Structure

```
deployment/
├── deployment/docker/          # Docker configuration
│   ├── docker-compose.yml      # Multi-service setup
│   ├── Dockerfile             # Container definition
│   ├── requirements.txt       # Python dependencies
│   ├── fastapi_app.py         # Backend API
│   └── gradio_app.py          # Web interface
├── models/                    # Model artifacts
│   ├── stable_diffusion_finetune_v1/  # LoRA adapter
│   └── [base model files]     # SD 1.5 model
└── start_backend.*            # Launch scripts
```

## 🚀 Production Deployment

For production use:

1. **GPU Instance**: Use cloud GPU instances (AWS P3, Google Cloud GPU)
2. **Load Balancer**: Add nginx or AWS ALB for multiple instances
3. **Monitoring**: Enable MLflow/TensorBoard for production monitoring
4. **Security**: Add authentication and rate limiting
5. **Scaling**: Use Kubernetes for auto-scaling

## 📞 Support

- Check logs: `docker-compose logs -f fastapi-backend`
- Health check: `curl http://localhost:8000/health`
- API docs: http://localhost:8000/docs
