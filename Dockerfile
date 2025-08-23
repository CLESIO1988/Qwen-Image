# Use standard PyTorch image which is guaranteed to exist
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

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

# Verify installations work
RUN python -c "import runpod; print('✅ runpod imported')"
RUN python -c "import torch; print('✅ torch imported:', torch.__version__)"
RUN python -c "import diffusers; print('✅ diffusers imported:', diffusers.__version__)"
RUN python -c "from diffusers import QwenImageEditPipeline; print('✅ QwenImageEditPipeline available')"

# Test CUDA availability
RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

CMD ["python", "handler.py"]
# Print versions for debugging
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
RUN python -c "import diffusers; print(f'Diffusers version: {diffusers.__version__}')"

CMD ["python", "handler.py"]RUN python -c "import runpod; import torch; import diffusers; print('All imports successful')"

# The container will be started by RunPod
# No CMD needed — RunPod uses its own entrypoint
