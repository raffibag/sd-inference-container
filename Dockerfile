FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Prevent timezone prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pin exact PyTorch versions for consistency
RUN pip3 install --no-cache-dir torch==2.1.2 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121

# Install Python packages for multi-LoRA inference with PyTorch 2.1.2 compatibility
RUN pip3 install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    peft \
    safetensors \
    opencv-python-headless \
    Pillow \
    numpy \
    flask \
    boto3

# Install xformers separately with PyTorch 2.1.2 compatibility
RUN pip3 install --no-cache-dir xformers --index-url https://download.pytorch.org/whl/cu121

# Copy inference scripts
COPY scripts/ /opt/ml/code/
WORKDIR /opt/ml/code

# SageMaker inference entry point
ENV SAGEMAKER_PROGRAM=inference.py

# Create model directory
RUN mkdir -p /opt/ml/model

EXPOSE 8080

ENTRYPOINT ["python", "inference.py"]