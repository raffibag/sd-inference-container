#!/usr/bin/env python3
"""
Enhanced SageMaker handler with ControlNet + LoRA support
Handles SDXL + multi-LoRA composition + ControlNet conditioning
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
import cv2
import numpy as np
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
controlnet_processors = {}

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
    """Initialize the Stable Diffusion XL pipeline with ControlNet support"""
    global pipe, s3_client, controlnet_processors
    
    try:
        from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
        from controlnet_aux import (
            OpenposeDetector, 
            CannyDetector, 
            MidasDetector,
            LineartDetector
        )
        import torch
        
        logger.info("🚀 Initializing SDXL + ControlNet pipeline...")
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Load ControlNet models
        logger.info("📦 Loading ControlNet models...")
        controlnet_openpose = ControlNetModel.from_pretrained(
            "diffusers/controlnet-openpose-sdxl-1.0",
            torch_dtype=torch.float16
        )
        
        controlnet_canny = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0", 
            torch_dtype=torch.float16
        )
        
        controlnet_depth = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            torch_dtype=torch.float16
        )
        
        # Store ControlNets in a dict for easy access
        controlnets = {
            'openpose': controlnet_openpose,
            'canny': controlnet_canny,
            'depth': controlnet_depth
        }
        
        # Load base SDXL model with first ControlNet (we'll swap as needed)
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_id,
            controlnet=controlnet_openpose,  # Default to openpose
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        
        # Store controlnets for swapping
        pipe.controlnets = controlnets
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        
        # Enable memory efficient attention
        if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
            pipe.enable_xformers_memory_efficient_attention()
        
        # Initialize ControlNet processors
        logger.info("🔧 Initializing ControlNet processors...")
        controlnet_processors = {
            'openpose': OpenposeDetector.from_pretrained('lllyasviel/openpose'),
            'canny': CannyDetector(),
            'depth': MidasDetector.from_pretrained('lllyasviel/midas'),
            'lineart': LineartDetector.from_pretrained('lllyasviel/lineart')
        }
        
        logger.info(f"✅ Pipeline initialized on {device} with ControlNet support")
        logger.info(f"🎛️  Available ControlNets: {list(controlnets.keys())}")
        logger.info(f"🔍 Available processors: {list(controlnet_processors.keys())}")
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

def process_control_image(image_data: str, control_type: str) -> Optional[Image.Image]:
    """Process control image using specified ControlNet processor"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        control_image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        processor = controlnet_processors.get(control_type)
        if not processor:
            logger.error(f"❌ Unknown control type: {control_type}")
            return None
        
        logger.info(f"🔍 Processing control image with {control_type}")
        
        if control_type == 'canny':
            # Convert PIL to OpenCV format for Canny
            opencv_image = cv2.cvtColor(np.array(control_image), cv2.COLOR_RGB2BGR)
            processed = processor(opencv_image)
            return Image.fromarray(processed)
        else:
            # Use processor directly
            processed = processor(control_image)
            return processed
        
    except Exception as e:
        logger.error(f"❌ Failed to process control image: {str(e)}")
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
    """SageMaker inference endpoint with ControlNet + LoRA support"""
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
        
        # ControlNet parameters
        control_image_data = input_data.get('control_image')  # base64 encoded
        control_type = input_data.get('control_type', 'openpose')  # openpose, canny, depth
        controlnet_conditioning_scale = input_data.get('controlnet_conditioning_scale', 1.0)
        
        logger.info(f"📝 Generating with prompt: {prompt}")
        logger.info(f"🎨 Composition: {json.dumps(composition)}")
        logger.info(f"🎛️  ControlNet: {control_type if control_image_data else 'None'}")
        
        # Apply LoRA composition if specified
        if composition:
            apply_lora_composition(composition)
        
        # Process control image if provided
        control_image = None
        if control_image_data and control_type in pipe.controlnets:
            control_image = process_control_image(control_image_data, control_type)
            if control_image:
                # Swap ControlNet if needed
                if pipe.controlnet != pipe.controlnets[control_type]:
                    pipe.controlnet = pipe.controlnets[control_type]
                logger.info(f"✅ Using {control_type} ControlNet with conditioning scale {controlnet_conditioning_scale}")
        
        # Generate images
        generator = torch.Generator(device=pipe.device).manual_seed(seed) if seed else None
        
        if control_image:
            # Generate with ControlNet
            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_images_per_prompt=num_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                width=width,
                height=height,
                generator=generator
            ).images
        else:
            # Generate without ControlNet (fallback to regular SDXL)
            from diffusers import StableDiffusionXLPipeline
            
            # Create temporary regular pipeline
            regular_pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            ).to(pipe.device)
            
            # Copy LoRA weights if they exist
            if hasattr(pipe, '_lora_scale'):
                regular_pipe.load_lora_weights = pipe.load_lora_weights
                regular_pipe.set_adapters = pipe.set_adapters
                if composition:
                    apply_lora_composition(composition)
            
            images = regular_pipe(
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
            'controlnet': {
                'type': control_type if control_image else None,
                'conditioning_scale': controlnet_conditioning_scale if control_image else None
            },
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
    
    logger.info("🚀 Starting Vibez ControlNet + LoRA Inference Server...")
    
    # Initialize pipeline
    if not initialize_pipeline():
        logger.error("❌ Failed to initialize pipeline, exiting...")
        sys.exit(1)
    
    # Start Flask server
    logger.info("🌐 Starting Flask server on port 8080...")
    app.run(host='0.0.0.0', port=8080, threaded=True)

if __name__ == '__main__':
    start_server()