#!/usr/bin/env python3
"""
SageMaker-compatible inference handler for Vibez Multi-LoRA
Implements proper health checks and inference endpoints
"""

import os
import json
import logging
import flask
import signal
import sys
# Import modules we need - avoid importing inference.py which has errors
import sys
import importlib.util

# Load just the functions we need from inference.py
spec = importlib.util.spec_from_file_location("inference_module", "/opt/ml/code/inference.py")
inference_module = importlib.util.module_from_spec(spec)

# Import the functions directly to avoid loading the broken class
from diffusers import StableDiffusionXLPipeline
import boto3 as boto3_module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = flask.Flask(__name__)

# Global composer instance
composer = None

def get_default_model_bucket():
    """Get default model bucket name based on environment"""
    try:
        import boto3
        
        # Get account ID and region from instance metadata or boto3
        session = boto3.Session()
        region = session.region_name or os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')
        
        # Try to get account ID
        try:
            sts = boto3.client('sts')
            account_id = sts.get_caller_identity()['Account']
        except:
            # Fallback to hardcoded if STS fails
            account_id = '796245059390'
        
        # Check environment
        env = os.environ.get('VIBEZ_ENV', 'prod')
        env_suffix = '' if env == 'prod' else f'-{env}'
        
        return f'vibez-model-registry{env_suffix}-{account_id}-{region}'
    except Exception as e:
        # Fallback to current bucket if AWS calls fail
        return 'vibez-model-registry-796245059390-us-west-2'

@app.route('/ping', methods=['GET'])
def ping():
    """SageMaker health check endpoint"""
    # Check if model is loaded
    if composer is None:
        return flask.Response(status=503, response="Model not loaded")
    
    # Optional: Check if base model is loaded
    if not hasattr(composer, 'pipe') or composer.pipe is None:
        return flask.Response(status=503, response="Pipeline not initialized")
    
    return flask.Response(status=200, response="OK")

@app.route('/invocations', methods=['POST'])
def invocations():
    """SageMaker inference endpoint"""
    if composer is None:
        return flask.Response(
            status=503,
            response=json.dumps({"error": "Model not loaded"}),
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
        
        # Extract composition and parameters
        composition = input_data.get('composition', {})
        prompt = input_data.get('prompt', '')
        negative_prompt = input_data.get('negative_prompt', '')
        num_images = input_data.get('num_images', 1)
        num_inference_steps = input_data.get('num_inference_steps', 50)
        guidance_scale = input_data.get('guidance_scale', 7.5)
        width = input_data.get('width', 1024)
        height = input_data.get('height', 1024)
        seed = input_data.get('seed', None)
        use_ai_prompts = input_data.get('use_ai_prompts', False)
        
        # Generate AI prompts if requested
        if use_ai_prompts:
            logger.info("🤖 Generating AI-optimized prompts...")
            prompt, negative_prompt = generate_ai_prompts(composition, prompt)
        
        logger.info(f"📝 Prompt: {prompt}")
        logger.info(f"🎨 Composition: {json.dumps(composition, indent=2)}")
        
        # Apply composition
        composer.apply_composition(composition)
        
        # Generate images
        import torch
        generator = torch.Generator(device=composer.device).manual_seed(seed) if seed else None
        
        images = composer.pipe(
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
        import base64
        from io import BytesIO
        
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
    global composer
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGTERM, sigterm_handler)
    
    logger.info("🚀 Starting Vibez Multi-LoRA Inference Server...")
    
    # Initialize model
    model_bucket = os.environ.get('MODEL_BUCKET', get_default_model_bucket())
    logger.info(f"📦 Using model bucket: {model_bucket}")
    
    try:
        composer = MultiLoRAComposer(model_bucket)
        logger.info("✅ Model initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {str(e)}")
        sys.exit(1)
    
    # Start Flask server
    logger.info("🌐 Starting Flask server on port 8080...")
    app.run(host='0.0.0.0', port=8080, threaded=True)

if __name__ == '__main__':
    start_server()