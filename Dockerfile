# Base image with CUDA 11.8, Python 3.10, and PyTorch stack
# Use the ultra-lean pytorch-base container as base
FROM pytorch-base:ultra-lean

# Install minimal system dependencies (NO ffmpeg from apt)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    ca-certificates \
    xz-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install ffmpeg via static binary (~25MB vs 1GB APT)
RUN curl -L -o /tmp/ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
    && mkdir -p /usr/local/bin/ffmpeg-static \
    && tar -xf /tmp/ffmpeg.tar.xz -C /usr/local/bin/ffmpeg-static --strip-components=1 \
    && ln -s /usr/local/bin/ffmpeg-static/ffmpeg /usr/local/bin/ffmpeg \
    && rm -rf /tmp/ffmpeg.tar.xz

# Python deps for inference and video generation
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
    fastapi \
    uvicorn[standard] \
    boto3 \
    imageio[ffmpeg] \
    scipy \
    numpy==1.26.4

# HuggingFace cache directories (avoid re-download)
ENV HF_HOME=/opt/ml/cache/huggingface
ENV TRANSFORMERS_CACHE=/opt/ml/cache/huggingface
RUN mkdir -p /opt/ml/cache/huggingface

# Copy inference code
COPY scripts/ /opt/ml/code/
WORKDIR /opt/ml/code

# Add serve script to PATH
RUN chmod +x /opt/ml/code/serve && ln -s /opt/ml/code/serve /usr/local/bin/serve

# Final cleanup to shrink image
RUN rm -rf /root/.cache /tmp/* /var/tmp/* /opt/ml/cache \
    && find /usr/local -depth -type d -name __pycache__ -exec rm -rf {} + \
    && find /usr/local -name '*.pyc' -delete

# Set SageMaker entrypoint
ENV SAGEMAKER_PROGRAM=inference.py

# Let SageMaker handle entrypoint
ENTRYPOINT []
CMD ["serve"]

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/ping || exit 1
