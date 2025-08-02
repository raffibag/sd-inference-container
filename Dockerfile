# Use AWS official PyTorch inference container - guaranteed SageMaker GPU compatibility
# This eliminates CUDA driver version mismatches on ml.g5.x instances
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.4.0-gpu-py311-cu124-ubuntu22.04-sagemaker

# Install our ML dependencies on top of AWS base image
RUN pip install --no-cache-dir \
    diffusers[torch]>=0.27.0 \
    transformers>=4.38.0 \
    accelerate>=0.25.0 \
    controlnet-aux>=0.0.12 \
    xformers>=0.0.26 \
    safetensors>=0.3.1 \
    opencv-python \
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