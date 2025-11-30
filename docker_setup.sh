#!/bin/bash

# Docker Setup Script for Stable Diffusion Fine-tuning
echo "🐳 Setting up Stable Diffusion Docker Environment"
echo "=================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if NVIDIA Docker is available (for GPU support)
if command -v nvidia-docker &> /dev/null; then
    echo "✅ NVIDIA Docker detected - GPU support available"
else
    echo "⚠️  NVIDIA Docker not detected - GPU support may not work"
    echo "Install NVIDIA Docker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

echo ""
echo "📦 Building Docker image..."
echo "This may take 10-20 minutes depending on your internet speed..."
echo ""

# Build the Docker image
start_time=$(date +%s)
docker build -t stable-diffusion-finetune .

end_time=$(date +%s)
build_time=$((end_time - start_time))

echo ""
echo "✅ Docker image built successfully!"
echo "Build time: $build_time seconds"
echo ""
echo "🚀 To run the container:"
echo "docker run --gpus all -it stable-diffusion-finetune"
echo ""
echo "🌐 To run with web interface:"
echo "docker-compose up gradio-interface"
echo ""
echo "📁 Your project files will be mounted as volumes"
echo "=================================================="
