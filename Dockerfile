FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System setup
RUN apt-get update && apt-get install -y \
    git wget curl python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Add this line to set the HF token as a build-time argument
ARG HF_AUTH_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=${HF_AUTH_TOKEN}

# Default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /app

# Copy dependency list (edit to match your repo)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy project code
COPY . .

# Run the RunPod handler
CMD ["python", "-u", "rp_handler.py"]
