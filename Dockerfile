# Start with PyTorch base image that includes CUDA 11.8
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# Set timezone to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# System dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN python -m pip install --upgrade pip setuptools wheel

# CRITICAL: Uninstall any existing NumPy and force install <2.0
RUN pip uninstall -y numpy && pip install --no-cache-dir "numpy<2.0"

# Core Python packages
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    xformers==0.0.23.post1 --extra-index-url https://download.pytorch.org/whl/cu118

# AI generation stack
RUN pip install --no-cache-dir \
    diffusers==0.27.2 \
    transformers==4.40.2 \
    accelerate==0.27.2 \
    huggingface-hub==0.20.3 \
    controlnet-aux==0.0.10 \
    peft==0.10.0 \
    safetensors==0.4.2 \
    einops==0.7.0 \
    triton==2.1.0 \
    scipy \
    opencv-python-headless \
    Pillow \
    flask \
    boto3 \
    imageio[ffmpeg] \
    moviepy

# Additional dependencies for ControlNet processors
RUN pip install --no-cache-dir \
    timm

# HuggingFace cache directories (avoid re-download)
ENV HF_HOME=/opt/ml/cache/huggingface
ENV TRANSFORMERS_CACHE=/opt/ml/cache/huggingface
RUN mkdir -p /opt/ml/cache/huggingface

# Copy inference code
COPY scripts/ /opt/ml/code/
WORKDIR /opt/ml/code

# Add serve script to path
RUN chmod +x /opt/ml/code/serve && ln -s /opt/ml/code/serve /usr/local/bin/serve

# Set environment variables for SageMaker
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV SAGEMAKER_PROGRAM=controlnet_lora_handler.py

# Let SageMaker handle entrypoint
ENTRYPOINT []