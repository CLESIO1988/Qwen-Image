# Use a lightweight PyTorch image with CUDA support
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set environment variables to reduce warnings and set cache location
ENV DEBIAN_FRONTEND=noninteractive
ENV TRANSFORMERS_CACHE=/tmp/cache
ENV HF_HOME=/tmp/cache
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install git (needed to install diffusers from GitHub)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Note: diffusers from GitHub includes QwenImageEditPipeline
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.3.0 torchvision pillow && \
    pip install --no-cache-dir git+https://github.com/huggingface/diffusers && \
    pip install --no-cache-dir transformers accelerate bitsandbytes numpy

# Copy the rest of the application code
COPY handler.py .

# Optional: Warm up the image (you can add more files here if needed)
# COPY . .

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import os; exit(0) if os.path.exists('/tmp/cache') else exit(1)"

# No CMD needed — RunPod uses its own entrypoint