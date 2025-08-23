# Use the same base image from your original pod.yaml
FROM runpod/pytorch:2.3.0-gpu-12.1

# Set environment variables to reduce warnings and set cache location
ENV DEBIAN_FRONTEND=noninteractive
ENV TRANSFORMERS_CACHE=/tmp/cache
ENV HF_HOME=/tmp/cache
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create cache directory
RUN mkdir -p /tmp/cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY handler.py .
COPY inference.py .

# Create outputs directory
RUN mkdir -p /tmp/outputs

# Test import to catch issues early
RUN python -c "import runpod; import torch; import diffusers; print('✅ All imports successful')"

# Test model import specifically
RUN python -c "from diffusers import QwenImageEditPipeline; print('✅ QwenImageEditPipeline import successful')"

# The container will be started by RunPod
