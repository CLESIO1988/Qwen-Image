# Use standard PyTorch image
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

# Basic import tests (skip QwenImageEditPipeline for now)
RUN python -c "import runpod; print('✅ runpod imported')"
RUN python -c "import torch; print('✅ torch imported:', torch.__version__)"
RUN python -c "import diffusers; print('✅ diffusers imported:', diffusers.__version__)"

# Test CUDA availability
RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Print versions for debugging
RUN python -c "import torch; import diffusers; print(f'PyTorch: {torch.__version__}, Diffusers: {diffusers.__version__}')"

# Create a test script to check QwenImageEditPipeline availability at runtime
RUN echo 'try:\n    from diffusers import QwenImageEditPipeline\n    print("✅ QwenImageEditPipeline available")\nexcept Exception as e:\n    print(f"❌ QwenImageEditPipeline not available: {e}")' > /app/test_qwen.py

RUN python /app/test_qwen.py
# Test CUDA availability
RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

CMD ["python", "handler.py"]
# Print versions for debugging
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
RUN python -c "import diffusers; print(f'Diffusers version: {diffusers.__version__}')"

CMD ["python", "handler.py"]RUN python -c "import runpod; import torch; import diffusers; print('All imports successful')"

# The container will be started by RunPod
# No CMD needed — RunPod uses its own entrypoint
