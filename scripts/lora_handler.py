#!/usr/bin/env python3
"""
Optimized SageMaker handler with LoRA support
Handles SDXL + multi-LoRA composition for Vibez
"""

import os
import json
import logging
import flask
import signal
import sys
import boto3
import torch
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, List, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = flask.Flask(__name__)

# Global instances
pipe = None
s3_client = None
lora_cache = {}

def get_default_model_bucket():
    """Get default model bucket name based on environment"""
    try:
        session = boto3.Session()
        region = session.region_name or os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')
        
        try:
            sts = boto3.client('sts')
            account_id = sts.get_caller_identity()['Account']
        except:
            account_id = '796245059390'
        
        env = os.environ.get('VIBEZ_ENV', 'prod')
        env_suffix = '' if env == 'prod' else f'-{env}'
        
        return f'vibez-model-registry{env_suffix}-{account_id}-{region}'
    except Exception as e:
        return 'vibez-model-registry-796245059390-us-west-2'

def initialize_pipeline():
    """Initialize the Stable Diffusion XL pipeline"""
    global pipe, s3_client
    
    try:
        from diffusers import StableDiffusionXLPipeline
        import torch
        
        logger.info("🚀 Initializing Stable Diffusion XL pipeline...")
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Load base SDXL model
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        
        # Enable memory efficient attention
        if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
            pipe.enable_xformers_memory_efficient_attention()
        
        logger.info(f"✅ Pipeline initialized on {device}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {str(e)}")
        return False

def download_lora_model(model_path: str, bucket: str) -> Optional[str]:
    """Download LoRA model from S3 if not cached"""
    cache_key = f"{bucket}/{model_path}"
    
    if cache_key in lora_cache:
        logger.info(f"✅ Using cached LoRA: {cache_key}")
        return lora_cache[cache_key]
    
    try:
        # Create local cache directory
        cache_dir = Path("/tmp/lora_cache")
        cache_dir.mkdir(exist_ok=True)
        
        # Download from S3
        local_path = cache_dir / model_path.replace('/', '_')
        local_path = local_path.with_suffix('.safetensors')
        
        logger.info(f"📦 Downloading LoRA from s3://{bucket}/{model_path}")
        s3_client.download_file(bucket, f"{model_path}/model.safetensors", str(local_path))
        
        lora_cache[cache_key] = str(local_path)
        return str(local_path)
        
    except Exception as e:
        logger.error(f"❌ Failed to download LoRA: {str(e)}")
        return None

def apply_lora_composition(composition: Dict):
    """Apply LoRA composition to the pipeline"""
    global pipe
    
    try:
        bucket = get_default_model_bucket()
        
        # Reset adapters first
        try:
            pipe.unload_lora_weights()
        except:
            pass
        
        adapters = []
        weights = []
        
        # Load character LoRA if specified
        if 'character' in composition:
            char_config = composition['character']
            model_path = char_config['model_path']
            weight = char_config.get('weight', 1.0)
            
            lora_path = download_lora_model(model_path, bucket)
            if lora_path:
                logger.info(f"🎭 Loading character LoRA with weight {weight}")
                pipe.load_lora_weights(lora_path, adapter_name="character")
                adapters.append("character")
                weights.append(weight)
        
        # Load style LoRA if specified
        if 'style' in composition:
            style_config = composition['style']
            model_path = style_config['model_path']
            weight = style_config.get('weight', 0.7)
            
            lora_path = download_lora_model(model_path, bucket)
            if lora_path:
                logger.info(f"🎨 Loading style LoRA with weight {weight}")
                pipe.load_lora_weights(lora_path, adapter_name="style")
                adapters.append("style")
                weights.append(weight)
        
        # Apply adapters if any were loaded
        if adapters:
            pipe.set_adapters(adapters, adapter_weights=weights)
            logger.info(f"✅ Applied LoRA composition: {adapters} with weights {weights}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to apply LoRA composition: {str(e)}")
        return False

@app.route('/ping', methods=['GET'])
def ping():
    """SageMaker health check endpoint"""
    if pipe is None:
        return flask.Response(status=503, response="Pipeline not loaded")
    
    return flask.Response(status=200, response="OK")

@app.route('/invocations', methods=['POST'])
def invocations():
    """SageMaker inference endpoint with LoRA support"""
    if pipe is None:
        return flask.Response(
            status=503,
            response=json.dumps({"error": "Pipeline not loaded"}),
            mimetype='application/json'
        )
    
    try:
        # Parse request
        content_type = flask.request.content_type
        if content_type == 'application/json':
            input_data = flask.request.json
        else:
            return flask.Response(
                status=400,
                response=json.dumps({"error": f"Unsupported content type: {content_type}"}),
                mimetype='application/json'
            )
        
        # Extract parameters
        composition = input_data.get('composition', {})
        prompt = input_data.get('prompt', 'a beautiful landscape')
        negative_prompt = input_data.get('negative_prompt', 'blurry, low quality, bad anatomy')
        num_images = input_data.get('num_images', 1)
        num_inference_steps = input_data.get('num_inference_steps', 30)
        guidance_scale = input_data.get('guidance_scale', 7.5)
        width = input_data.get('width', 1024)
        height = input_data.get('height', 1024)
        seed = input_data.get('seed', None)
        
        logger.info(f"📝 Generating with prompt: {prompt}")
        logger.info(f"🎨 Composition: {json.dumps(composition)}")
        
        # Apply LoRA composition if specified
        if composition:
            apply_lora_composition(composition)
        
        # Generate images
        generator = torch.Generator(device=pipe.device).manual_seed(seed) if seed else None
        
        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        ).images
        
        # Convert to base64
        image_data = []
        for img in images:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_data.append(img_base64)
        
        response = {
            'images': image_data,
            'composition': composition,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'parameters': {
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'width': width,
                'height': height,
                'seed': seed
            }
        }
        
        logger.info(f"✅ Generated {len(image_data)} image(s)")
        
        return flask.Response(
            status=200,
            response=json.dumps(response),
            mimetype='application/json'
        )
        
    except Exception as e:
        logger.error(f"❌ Inference error: {str(e)}")
        return flask.Response(
            status=500,
            response=json.dumps({"error": str(e)}),
            mimetype='application/json'
        )

def sigterm_handler(signum, frame):
    """Handle graceful shutdown"""
    logger.info("Received SIGTERM, shutting down gracefully...")
    sys.exit(0)

def start_server():
    """Start the Flask server with proper initialization"""
    global pipe
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGTERM, sigterm_handler)
    
    logger.info("🚀 Starting Vibez LoRA Inference Server...")
    
    # Initialize pipeline
    if not initialize_pipeline():
        logger.error("❌ Failed to initialize pipeline, exiting...")
        sys.exit(1)
    
    # Start Flask server
    logger.info("🌐 Starting Flask server on port 8080...")
    app.run(host='0.0.0.0', port=8080, threaded=True)

if __name__ == '__main__':
    start_server()