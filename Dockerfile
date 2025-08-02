# Use PyTorch base image with CUDA 11.8 (matching training container packages)
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# Install system dependencies including OpenGL for cv2
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Force downgrade NumPy to fix PyTorch compatibility
RUN pip install --force-reinstall "numpy<2.0" 

# Install Python dependencies
RUN pip install --no-cache-dir \
    diffusers==0.27.2 \
    transformers==4.40.2 \
    accelerate==0.27.2 \
    safetensors==0.4.2 \
    controlnet-aux==0.0.10 \
    opencv-python-headless \
    peft==0.10.0 \
    Pillow \
    flask \
    boto3

# Force reinstall PyTorch with proper CUDA 11.8 support (after other deps to avoid conflicts)
RUN pip install --force-reinstall --no-cache-dir torch==2.1.2 torchvision==0.16.2 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install xformers with CUDA 11.8 support (compatible with PyTorch 2.1.2)
RUN pip install xformers==0.0.23.post1 --extra-index-url https://download.pytorch.org/whl/cu118

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

# Remove direct entrypoint to let SageMaker handle it properly
ENTRYPOINT []