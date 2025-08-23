# Use RunPod's recommended base image
FROM runpod/pytorch:2.3.0-py3.11-cuda12.1.0-devel-ubuntu22.04

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

# Copy the handler
COPY handler.py .

# Create outputs directory
RUN mkdir -p /app/outputs

# Test import to catch issues early
RUN python -c "import runpod; import torch; import diffusers; print('All imports successful')"

# The container will be started by RunPod
# No CMD needed — RunPod uses its own entrypoint
