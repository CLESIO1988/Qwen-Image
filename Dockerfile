FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install Git and other essential system dependencies
RUN apt-get update && apt-get install -y \
    git wget curl python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Set default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1


# Add this line to set the HF token as a build-time argument
ARG HF_AUTH_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=${HF_AUTH_TOKEN}

# Set the working directory
WORKDIR /app

# Copy and install Python dependencies. This step will now succeed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install RunPod serverless SDK
RUN pip install runpod

# Copy project code
COPY . .

# Run the RunPod handler
CMD ["python", "-u", "rp_handler.py"]