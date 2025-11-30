# Stable Diffusion Fine-tuning Docker Environment
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (Python is already included in PyTorch image)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Create workspace
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for models and outputs
RUN mkdir -p models lora_adapter original_images finetuned_images combined_dataset enhanced_dataset

# Download and cache the base model during build (optional)
# RUN python -c "from diffusers import StableDiffusionPipeline; pipe = StableDiffusionPipeline.from_pretrained('sd-legacy/stable-diffusion-v1-5'); print('Model cached')"

# Expose port for potential web interface
EXPOSE 7860

# Default command
CMD ["python", "--version"]
