# 🔧 Troubleshooting Guide

Common issues and solutions for Stable Diffusion deployment.

## 🚨 Critical Issues

### Model Not Loading

**Symptoms:**
- API returns 503 Service Unavailable
- "Model not loaded" error messages
- Health check shows `"loaded": false`

**Solutions:**

1. **Check GPU availability:**
   ```bash
   nvidia-smi
   # Should show your GPU and memory usage
   ```

2. **Verify CUDA installation:**
   ```bash
   nvcc --version
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Check model file paths:**
   ```bash
   ls -la models/
   ls -la models/stable_diffusion_finetune_v1/
   ```

4. **Increase Docker memory limits:**
   ```bash
   # In docker-compose.yml, add:
   deploy:
     resources:
       limits:
         memory: 16G
   ```

5. **Restart the service:**
   ```bash
   docker-compose down
   docker-compose up --build
   ```

### CUDA Out of Memory

**Symptoms:**
- "CUDA out of memory" errors
- Generation requests fail
- GPU memory usage at 100%

**Solutions:**

1. **Reduce batch size and resolution:**
   ```json
   {
     "width": 256,
     "height": 256,
     "num_images": 1
   }
   ```

2. **Enable memory optimization:**
   ```python
   # Add to fastapi_app.py
   torch.cuda.empty_cache()
   ```

3. **Use gradient checkpointing:**
   ```python
   pipe.enable_attention_slicing()
   ```

4. **Restart GPU memory:**
   ```bash
   sudo fuser -v /dev/nvidia* | awk '{print $2}' | xargs kill -9
   ```

## 🔧 API Issues

### Connection Refused

**Symptoms:**
- Cannot connect to localhost:8000
- curl: (7) Failed to connect

**Solutions:**

1. **Check if service is running:**
   ```bash
   docker-compose ps
   # Should show stable-diffusion-api as "Up"
   ```

2. **Check port availability:**
   ```bash
   netstat -tlnp | grep 8000
   ```

3. **Check container logs:**
   ```bash
   docker-compose logs stable-diffusion-api
   ```

4. **Restart service:**
   ```bash
   docker-compose restart stable-diffusion-api
   ```

### Slow Response Times

**Symptoms:**
- Generation takes >10 seconds
- API feels sluggish

**Solutions:**

1. **Check GPU utilization:**
   ```bash
   nvidia-smi
   # GPU should be actively used during generation
   ```

2. **Optimize PyTorch:**
   ```python
   torch.backends.cudnn.benchmark = True
   torch.backends.cuda.matmul.allow_tf32 = True
   ```

3. **Use faster inference settings:**
   ```json
   {
     "num_inference_steps": 15,
     "guidance_scale": 7.0
   }
   ```

4. **Check system resources:**
   ```bash
   top  # CPU usage
   free -h  # Memory usage
   ```

## 🐳 Docker Issues

### Build Failures

**Symptoms:**
- `docker-compose build` fails
- "No such file or directory" errors

**Solutions:**

1. **Check Docker context:**
   ```bash
   ls -la deployment/docker/
   # Ensure Dockerfile and requirements are present
   ```

2. **Clear Docker cache:**
   ```bash
   docker system prune -a
   ```

3. **Build with no cache:**
   ```bash
   docker-compose build --no-cache
   ```

4. **Check disk space:**
   ```bash
   df -h
   # Ensure >20GB free space
   ```

### GPU Not Accessible in Container

**Symptoms:**
- Container runs but model loads on CPU
- "CUDA not available" in logs

**Solutions:**

1. **Install NVIDIA Docker:**
   ```bash
   # Ubuntu
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Test GPU access:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

3. **Update docker-compose.yml:**
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

## 📊 Monitoring Issues

### MLflow Not Tracking

**Symptoms:**
- No experiments appear in MLflow UI
- Tracking calls don't show errors but no data

**Solutions:**

1. **Check MLflow server:**
   ```bash
   mlflow server --backend-store-uri ./mlruns --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000
   ```

2. **Verify tracking URI:**
   ```python
   import mlflow
   mlflow.set_tracking_uri("http://localhost:5000")
   ```

3. **Check permissions:**
   ```bash
   chmod -R 755 mlruns/
   ```

### TensorBoard Not Showing Data

**Symptoms:**
- TensorBoard starts but shows no data
- Empty graphs and metrics

**Solutions:**

1. **Check log directory:**
   ```bash
   ls -la tensorboard_logs/
   ```

2. **Start TensorBoard correctly:**
   ```bash
   tensorboard --logdir ./tensorboard_logs --host 0.0.0.0 --port 6006
   ```

3. **Flush writer before viewing:**
   ```python
   monitor.flush()  # In your code
   ```

## 🔒 Security Issues

### Unauthorized Access

**Symptoms:**
- API accepts requests without authentication
- No rate limiting working

**Solutions:**

1. **Enable authentication:**
   ```bash
   export API_KEY=your-secret-key
   # Restart service
   ```

2. **Add rate limiting middleware**
3. **Use HTTPS in production**
4. **Configure firewall rules**

## 🚀 Performance Issues

### Memory Leaks

**Symptoms:**
- Memory usage increases over time
- Eventually runs out of memory

**Solutions:**

1. **Add memory cleanup:**
   ```python
   import gc
   torch.cuda.empty_cache()
   gc.collect()
   ```

2. **Restart service periodically**
3. **Monitor memory usage:**
   ```bash
   watch -n 1 'nvidia-smi --query-gpu=memory.used --format=csv'
   ```

### High Latency

**Symptoms:**
- Response times >5 seconds consistently

**Solutions:**

1. **Profile the code:**
   ```python
   import cProfile
   cProfile.run('generate_image(...)')
   ```

2. **Use async processing:**
   ```python
   # Implement background task processing
   @app.post("/generate")
   async def generate_image(request: GenerationRequest, background_tasks: BackgroundTasks):
       background_tasks.add_task(process_generation, request)
       return {"status": "processing"}
   ```

3. **Optimize model loading:**
   ```python
   pipe = pipe.to("cuda")
   pipe.enable_xformers_memory_efficient_attention()
   ```

## 📝 Logging and Debugging

### Enable Debug Logging

```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or in docker-compose.yml
environment:
  - LOG_LEVEL=DEBUG
```

### Check Logs

```bash
# Docker logs
docker-compose logs -f stable-diffusion-api

# Application logs (if mounted)
tail -f logs/api.log

# System logs
journalctl -u docker -f
```

### Debug Commands

```bash
# Test model loading manually
python -c "
from diffusers import StableDiffusionPipeline
import torch
pipe = StableDiffusionPipeline.from_pretrained('sd-legacy/stable-diffusion-v1-5')
print('Model loaded successfully')
"

# Test GPU memory
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB')
"
```

## 🔄 Recovery Procedures

### Emergency Restart

```bash
# Force restart everything
docker-compose down -v
docker system prune -f
docker-compose up --build --force-recreate
```

### Data Recovery

```bash
# Restore from backup
tar -xzf models_backup_20240115.tar.gz -C ./

# Restore MLflow data
cp mlruns_backup.db mlruns.db
```

### Clean Reset

```bash
# Remove all containers and volumes
docker-compose down -v --remove-orphans

# Remove images
docker-compose down --rmi all

# Clean build
docker system prune -f
docker-compose up --build
```

## 📞 Getting Help

If these solutions don't work:

1. **Collect diagnostic information:**
   ```bash
   # System info
   uname -a
   nvidia-smi
   docker --version
   python --version

   # Container info
   docker-compose ps
   docker-compose logs > debug_logs.txt
   ```

2. **Check GitHub issues** for similar problems

3. **Create a detailed bug report** with:
   - Full error messages
   - System specifications
   - Docker configuration
   - Steps to reproduce

4. **Include relevant logs** and configuration files

---

**Remember:** Most issues are related to GPU access, memory limits, or incorrect file paths. Start with the basics and work your way up!

