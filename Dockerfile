# Use AWS official PyTorch inference container - guaranteed SageMaker GPU compatibility
# This eliminates CUDA driver version mismatches on ml.g5.x instances
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.4.0-gpu-py311-cu124-ubuntu22.04-sagemaker

# Prevent timezone prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set HuggingFace cache to avoid stalls
ENV TRANSFORMERS_CACHE=/opt/ml/cache/huggingface
ENV HF_HOME=/opt/ml/cache/huggingface

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# PyTorch 2.4.0 + CUDA 12.4 already included in AWS base image - no manual installation needed

# Install numpy first as it's a critical dependency (compatible with PyTorch 2.4.0)
RUN pip3 install --no-cache-dir "numpy>=1.25.2" \
    && pip3 cache purge

# Install all other dependencies from requirements.txt (they will skip torch/torchvision as already satisfied)
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && pip3 cache purge

# Install xformers with PyTorch 2.4.0 + CUDA 12.4 compatibility (AWS SageMaker optimized)
RUN pip3 install --no-cache-dir xformers>=0.0.26 --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 cache purge

# Copy inference scripts
COPY scripts/ /opt/ml/code/
WORKDIR /opt/ml/code

# SageMaker configuration
ENV SAGEMAKER_PROGRAM=controlnet_lora_handler.py
# HuggingFace token will be passed as environment variable during deployment
RUN mkdir -p /opt/ml/model
RUN mkdir -p /opt/ml/cache/huggingface

EXPOSE 8080
ENTRYPOINT ["python", "controlnet_lora_handler.py"]