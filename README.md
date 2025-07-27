# Stable Diffusion Inference Container

Production-ready Docker container for Stable Diffusion SDXL inference with Multi-LoRA composition and Claude-powered prompt optimization.

## Features

- **SDXL Multi-LoRA**: Load and compose multiple LoRA models simultaneously
- **Claude Integration**: AI-powered prompt optimization via AWS Bedrock
- **Production Ready**: SageMaker endpoint integration with Flask API
- **Memory Optimized**: xformers + model CPU offloading for efficiency
- **Latest Stack**: PyTorch 2.4.0 + CUDA 12.4 for maximum performance

## Container Contents

- PyTorch 2.4.0 with CUDA 12.4 support
- SDXL base model (1024x1024 native resolution)
- Multi-LoRA composition engine
- Claude 3.5 Sonnet integration for prompt generation
- Flask API for SageMaker endpoints
- Memory-optimized inference pipeline

## Build with CodeBuild

This container is built using AWS CodeBuild for reliable, high-speed builds and ECR deployment.

**Target ECR:** `796245059390.dkr.ecr.us-west-2.amazonaws.com/stable-diffusion-inference:latest`

## API Endpoints

### Health Check
```
GET /ping
```

### Image Generation
```
POST /invocations
```

**Request Format:**
```json
{
  "composition": {
    "character": {"model_path": "character/franka/20240127-1423", "weight": 1.0},
    "style": {"model_path": "style/sinisha/20240127-1445", "weight": 0.8}
  },
  "character_description": "Epic fantasy warrior",
  "use_ai_prompts": true,
  "num_inference_steps": 30,
  "guidance_scale": 7.5,
  "width": 1024,
  "height": 1024,
  "num_images": 1,
  "seed": 42
}
```

## Multi-LoRA Composition

The container can dynamically load and blend multiple LoRA models:

- **Character LoRAs**: Person identity and facial features
- **Style LoRAs**: Artistic style and photography techniques
- **Weighted Blending**: Adjust influence of each LoRA (0.0-1.0)
- **Intelligent Prompts**: Claude generates optimized prompts with trigger words

## Integration with Training Pipeline

1. **Training**: Kohya container creates LoRA models → S3 model registry
2. **Inference**: This container loads LoRAs → generates final images
3. **Composition**: Multiple LoRAs combined for unique character creation

## Performance Optimizations

- Latest xformers for memory efficiency
- Model CPU offloading for large models
- CUDA 12.4 optimizations
- Automatic mixed precision
- Memory-efficient attention mechanisms