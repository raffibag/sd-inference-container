# Use AWS official PyTorch inference container with CUDA 11.8 - confirmed compatible with ml.g5.2xlarge
# Matches SageMaker host driver ~525.60 to fix GPU detection
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.4.0-gpu-py310-cu118-ubuntu20.04-sagemaker

# Install our ML dependencies with exact versions for CUDA 11.8 compatibility
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.17.0 \
    xformers==0.0.27.post1 --extra-index-url https://download.pytorch.org/whl/cu118 \
    diffusers==0.27.2 \
    transformers==4.40.2 \
    accelerate==0.27.2 \
    safetensors==0.4.2 \
    controlnet-aux==0.0.12 \
    opencv-python \
    peft==0.10.0 \
    triton==2.0.0 \
    Pillow \
    numpy>=1.25.2 \
    flask \
    boto3

# Set HuggingFace cache to avoid stalls
ENV TRANSFORMERS_CACHE=/opt/ml/cache/huggingface
ENV HF_HOME=/opt/ml/cache/huggingface

# Copy inference scripts
COPY scripts/ /opt/ml/code/
WORKDIR /opt/ml/code

# SageMaker configuration - use our custom ControlNet + LoRA handler
ENV SAGEMAKER_PROGRAM=controlnet_lora_handler.py

# Create cache directory
RUN mkdir -p /opt/ml/cache/huggingface