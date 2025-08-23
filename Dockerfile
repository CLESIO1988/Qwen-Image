# Use the exact same base image from your pod.yaml that was working
FROM runpod/pytorch:2.3.0-gpu-12.1

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TRANSFORMERS_CACHE=/tmp/cache
ENV HF_HOME=/tmp/cache
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create cache directories
RUN mkdir -p /tmp/cache && mkdir -p /tmp/outputs

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY handler.py .
COPY inference.py .

# Verify installations
RUN python -c "import runpod; print('✅ runpod imported')"
RUN python -c "import torch; print('✅ torch imported')"
RUN python -c "import diffusers; print('✅ diffusers imported')"
RUN python -c "from diffusers import QwenImageEditPipeline; print('✅ QwenImageEditPipeline available')"

# Print versions for debugging
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
RUN python -c "import diffusers; print(f'Diffusers version: {diffusers.__version__}')"

CMD ["python", "handler.py"]RUN python -c "import runpod; import torch; import diffusers; print('All imports successful')"

# The container will be started by RunPod
# No CMD needed — RunPod uses its own entrypoint
