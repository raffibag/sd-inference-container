FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

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

# Install Python packages for multi-LoRA inference with latest versions
RUN pip3 install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    xformers --index-url https://download.pytorch.org/whl/cu124 \
    peft \
    safetensors \
    opencv-python \
    Pillow \
    numpy \
    flask \
    boto3

# Copy inference scripts
COPY scripts/ /opt/ml/code/
WORKDIR /opt/ml/code

# SageMaker inference entry point
ENV SAGEMAKER_PROGRAM inference.py

# Create model directory
RUN mkdir -p /opt/ml/model

EXPOSE 8080

ENTRYPOINT ["python", "inference.py"]