FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Prevent timezone prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

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

# Install PyTorch with CUDA 11.8 to match training container exactly
RUN pip3 install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118 \
    && pip3 cache purge

# Install remaining dependencies from requirements.txt (excluding torch/torchvision)
RUN pip3 install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    peft \
    safetensors \
    Pillow \
    flask \
    boto3 \
    && pip3 cache purge

# Install xformers with CUDA 11.8 to match training container
RUN pip3 install --no-cache-dir xformers --index-url https://download.pytorch.org/whl/cu118 \
    && pip3 cache purge

# Copy inference scripts
COPY scripts/ /opt/ml/code/
WORKDIR /opt/ml/code

# SageMaker configuration
ENV SAGEMAKER_PROGRAM=lora_handler.py
RUN mkdir -p /opt/ml/model

EXPOSE 8080
ENTRYPOINT ["python", "lora_handler.py"]