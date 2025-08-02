# Use AWS official PyTorch inference container with older versions compatible with SageMaker drivers
# PyTorch 2.1.2 + CUDA 11.8 + driver 11040 compatibility
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.1.2-gpu-py310-cu118-ubuntu20.04-sagemaker

# Install exact versions compatible with SageMaker driver 11040
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    xformers==0.0.23.post1 --extra-index-url https://download.pytorch.org/whl/cu118 \
    diffusers==0.27.2 \
    transformers==4.40.2 \
    accelerate==0.27.2 \
    safetensors==0.4.2 \
    controlnet-aux==0.0.12 \
    opencv-python \
    peft==0.10.0 \
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