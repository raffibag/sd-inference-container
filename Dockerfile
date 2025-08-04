# Stage 1: Build base image
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel AS ml-base

# Set timezone to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Common system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 support
RUN pip install --force-reinstall --no-cache-dir torch==2.1.2 torchvision==0.16.2 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install xformers with CUDA 11.8 support
RUN pip install xformers --extra-index-url https://download.pytorch.org/whl/cu118

# Set common environment variables
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Create common directories
RUN mkdir -p /opt/ml/code /opt/ml/cache

WORKDIR /opt/ml/code

# Stage 2: Build inference image
FROM ml-base

# Additional system dependencies for inference
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install specific xformers version for inference
RUN pip install --no-cache-dir xformers==0.0.23.post1 --extra-index-url https://download.pytorch.org/whl/cu118

# AI generation stack - install normally with dependencies
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
    opencv-python-headless \
    Pillow \
    flask \
    boto3 \
    imageio[ffmpeg] \
    moviepy \
    timm \
    scipy

# CRITICAL: Force downgrade NumPy to 1.x as the very last step
# This ensures all dependencies are installed but we still get NumPy 1.x
RUN pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

# Verify NumPy version
RUN python -c "import numpy; assert numpy.__version__.startswith('1.'), f'NumPy {numpy.__version__} is not 1.x'"

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
ENV SAGEMAKER_PROGRAM=controlnet_lora_handler.py

# Let SageMaker handle entrypoint
ENTRYPOINT []
CMD ["serve"]

# Health check for better monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/ping || exit 1