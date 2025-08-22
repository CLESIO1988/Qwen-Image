FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System setup
RUN apt-get update && apt-get install -y \
    git wget curl python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /app

# Copy dependency list (edit to match your repo)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install RunPod serverless SDK
RUN pip install runpod

# Copy project code
COPY . .

# Run the RunPod handler
CMD ["python", "-u", "rp_handler.py"]
