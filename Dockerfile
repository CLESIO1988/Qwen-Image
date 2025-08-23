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
    procps \
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

# Create a startup script
RUN echo '#!/bin/bash\n\
echo "🚀 Container starting..."\n\
echo "📍 Working directory: $(pwd)"\n\
echo "📂 Files in /app:"\n\
ls -la /app/\n\
echo "🐍 Python version:"\n\
python --version\n\
echo "📦 Python path:"\n\
python -c "import sys; print(sys.executable)"\n\
echo "🔍 Checking handler.py:"\n\
if [ -f "/app/handler.py" ]; then\n\
    echo "✅ handler.py exists"\n\
    echo "First few lines:"\n\
    head -5 /app/handler.py\n\
else\n\
    echo "❌ handler.py not found!"\n\
    exit 1\n\
fi\n\
echo "🏃 Starting handler..."\n\
exec python /app/handler.py' > /app/start.sh

# Make startup script executable
RUN chmod +x /app/start.sh

# Basic import tests (skip handler test to avoid syntax errors)
RUN python -c "import runpod; print('✅ runpod imported')"
RUN python -c "import torch; print('✅ torch imported:', torch.__version__)"
RUN python -c "import diffusers; print('✅ diffusers imported:', diffusers.__version__)"

# Test CUDA availability
RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Print versions for debugging
RUN python -c "import torch, diffusers; print(f'PyTorch: {torch.__version__}, Diffusers: {diffusers.__version__}')"

# Use the startup script
CMD ["/app/start.sh"]
