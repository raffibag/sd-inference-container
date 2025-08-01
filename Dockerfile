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

# Install PyTorch first with CUDA 11.8 index
RUN pip3 install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118 \
    && pip3 cache purge

# Install all other dependencies from requirements.txt (they will skip torch/torchvision as already satisfied)
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && pip3 cache purge

# Install xformers with specific version for CUDA 11.8
RUN pip3 install --no-cache-dir xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118 \
    && pip3 cache purge

# Copy inference scripts
COPY scripts/ /opt/ml/code/
WORKDIR /opt/ml/code

# SageMaker configuration
ENV SAGEMAKER_PROGRAM=controlnet_lora_handler.py
# HuggingFace token will be passed as environment variable during deployment
RUN mkdir -p /opt/ml/model

EXPOSE 8080
ENTRYPOINT ["python", "controlnet_lora_handler.py"]