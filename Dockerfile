# Use PyTorch base image with CUDA 12.1 (same as training container)
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies first
RUN pip install --no-cache-dir \
    diffusers==0.27.2 \
    transformers==4.40.2 \
    accelerate==0.27.2 \
    safetensors==0.4.2 \
    controlnet-aux==0.0.10 \
    opencv-python \
    peft==0.10.0 \
    Pillow \
    numpy>=1.25.2 \
    flask \
    boto3

# Force reinstall PyTorch with proper CUDA 12.1 support (after other deps to avoid conflicts)
RUN pip install --force-reinstall --no-cache-dir torch==2.1.2 torchvision==0.16.2 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install xformers with CUDA 12.1 support (compatible with PyTorch 2.1.2)
RUN pip install xformers==0.0.23.post1 --extra-index-url https://download.pytorch.org/whl/cu121

# Set HuggingFace cache to avoid stalls
ENV TRANSFORMERS_CACHE=/opt/ml/cache/huggingface
ENV HF_HOME=/opt/ml/cache/huggingface

# Copy inference scripts
COPY scripts/ /opt/ml/code/
WORKDIR /opt/ml/code

# Make serve script executable and add to PATH
RUN chmod +x /opt/ml/code/serve && ln -s /opt/ml/code/serve /usr/local/bin/serve

# Set environment variables
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV SAGEMAKER_PROGRAM=controlnet_lora_handler.py

# Create cache directory
RUN mkdir -p /opt/ml/cache/huggingface

# SageMaker expects 'serve' command - keep default entrypoint
# CMD will be overridden by SageMaker with 'serve'