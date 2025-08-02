#!/usr/bin/env python3
"""
Enhanced SageMaker handler with ControlNet + LoRA support
Handles SDXL + multi-LoRA composition + ControlNet conditioning
"""

import os
import json
import logging
import sys

# Configure logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug: Check Python environment and numpy availability
logger.info(f"Python executable: {sys.executable}")
logger.info(f"Python version: {sys.version}")

try:
    import numpy
    logger.info(f"✅ NumPy version: {numpy.__version__}")
except ImportError as e:
    logger.error(f"❌ NumPy not found in runtime environment: {e}")
    # Try to list installed packages
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
        logger.info(f"Installed packages:\n{result.stdout}")
    except Exception as ex:
        logger.error(f"Could not list packages: {ex}")

import flask
import signal
import boto3
import torch
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from typing import Dict, List, Optional
from pathlib import Path

# Initialize Flask app
app = flask.Flask(__name__)

# Global instances
pipe = None  # Regular SDXL pipeline
controlnet_pipe = None  # ControlNet pipeline
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
    """Initialize both regular SDXL and ControlNet pipelines"""
    global pipe, controlnet_pipe, s3_client, controlnet_processors
    
    try:
        from diffusers import StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, ControlNetModel
        from register_controlnets import register_controlnets
        import torch
        import os
        
        # Set HuggingFace cache directory
        os.environ["TRANSFORMERS_CACHE"] = "/opt/ml/cache/huggingface"
        os.environ["HF_HOME"] = "/opt/ml/cache/huggingface"
        
        logger.info("🚀 Initializing SDXL pipelines...")
        
        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 Cleared GPU cache")
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Check device first to determine dtype
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        logger.info(f"🖥️  Device: {device}, dtype: {dtype}")
        
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        
        # 1. Load regular SDXL pipeline (for non-ControlNet generation)
        logger.info("📦 Loading regular SDXL pipeline...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
            use_auth_token=hf_token if hf_token else None
        )
        
        # Disable safety checker to prevent freezing
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        
        # 2. Load base ControlNet pipeline  
        logger.info("📦 Loading base ControlNet pipeline...")
        
        # Get HuggingFace token from environment
        hf_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
        if hf_token:
            logger.info("🔑 Using HuggingFace token for model downloads")
        
        # Create initial ControlNet pipeline with canny (will be extended)
        initial_controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0-small", 
            torch_dtype=dtype,
            use_auth_token=hf_token
        )
        
        controlnet_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_id,
            controlnet=initial_controlnet,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
            use_auth_token=hf_token if hf_token else None
        )
        
        # Disable safety checker to prevent freezing
        controlnet_pipe.safety_checker = None
        controlnet_pipe.requires_safety_checker = False
        
        # 3. Register all ControlNets using the new system
        logger.info("📦 Registering ControlNet models and processors...")
        controlnet_config = {
            "canny": "diffusers/controlnet-canny-sdxl-1.0-small",
            "depth": "diffusers/controlnet-depth-sdxl-1.0",
            "openpose": "thibaud/controlnet-openpose-sdxl-1.0"
        }
        
        # Register all ControlNets and processors
        controlnets, controlnet_processors = register_controlnets(controlnet_pipe, controlnet_config)
        
        # Make controlnet_processors available globally
        globals()['controlnet_processors'] = controlnet_processors
        
        # Set default ControlNet
        controlnet_pipe.controlnet = controlnets["canny"]
        
        # Move both pipelines to device
        pipe = pipe.to(device)
        controlnet_pipe = controlnet_pipe.to(device)
        
        logger.info(f"🎮 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"🎮 CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Enable memory optimizations
        if device == "cuda":
            for p in [pipe, controlnet_pipe]:
                # Enable xformers memory efficient attention
                if hasattr(p, 'enable_xformers_memory_efficient_attention'):
                    try:
                        p.enable_xformers_memory_efficient_attention()
                        logger.info(f"✅ Enabled xformers for {p.__class__.__name__}")
                    except Exception as e:
                        logger.warning(f"⚠️  Could not enable xformers: {str(e)}")
                
                # Enable attention slicing for memory efficiency
                if hasattr(p, 'enable_attention_slicing'):
                    try:
                        p.enable_attention_slicing()
                        logger.info(f"✅ Enabled attention slicing for {p.__class__.__name__}")
                    except Exception as e:
                        logger.warning(f"⚠️  Could not enable attention slicing: {str(e)}")
                
                # Enable model CPU offload for large models
                if hasattr(p, 'enable_model_cpu_offload'):
                    try:
                        p.enable_model_cpu_offload()
                        logger.info(f"✅ Enabled CPU offload for {p.__class__.__name__}")
                    except Exception as e:
                        logger.warning(f"⚠️  Could not enable CPU offload: {str(e)}")
        else:
            logger.info("🖥️  Running on CPU - memory optimizations disabled")
        
        
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
        
        # Try different possible paths for the model
        date_part = model_path.split('/')[-1].replace('auto-', '')  # Remove auto- prefix
        possible_paths = [
            f"models/{model_path}/model.safetensors",
            f"{model_path}/model.safetensors",
            # Handle the tar.gz structure we found - with char-franka prefix and date (without auto-)
            f"models/{model_path}/char-franka-{date_part}/output/model.tar.gz",
            f"models/{model_path}/char-{date_part}/output/model.tar.gz",
            f"models/{model_path}/{date_part}/output/model.tar.gz"
        ]
        
        local_path = None
        for s3_path in possible_paths:
            try:
                local_file = cache_dir / f"{model_path.replace('/', '_')}_model"
                if s3_path.endswith('.tar.gz'):
                    # Download and extract tar.gz
                    tar_path = local_file.with_suffix('.tar.gz')
                    logger.info(f"📦 Trying to download LoRA from s3://{bucket}/{s3_path}")
                    s3_client.download_file(bucket, s3_path, str(tar_path))
                    
                    # Extract the tar.gz to find the safetensors file
                    import tarfile
                    extract_dir = cache_dir / f"{model_path.replace('/', '_')}_extracted"
                    extract_dir.mkdir(exist_ok=True)
                    
                    with tarfile.open(tar_path, 'r:gz') as tar:
                        tar.extractall(extract_dir)
                    
                    # Find the main safetensors file (look for the base model, not checkpoints)
                    main_model_candidates = ['franka_lora.safetensors', 'model.safetensors', 'pytorch_lora_weights.safetensors']
                    for candidate in main_model_candidates:
                        candidate_path = extract_dir / candidate
                        if candidate_path.exists():
                            local_path = candidate_path
                            logger.info(f"✅ Found main LoRA model: {candidate}")
                            break
                    
                    # Fallback: use any safetensors file if main model not found
                    if not local_path:
                        for file in extract_dir.rglob('*.safetensors'):
                            local_path = file
                            logger.info(f"✅ Found LoRA model (fallback): {file.name}")
                            break
                    
                    if local_path:
                        break
                else:
                    local_path = local_file.with_suffix('.safetensors')
                    logger.info(f"📦 Trying to download LoRA from s3://{bucket}/{s3_path}")
                    s3_client.download_file(bucket, s3_path, str(local_path))
                    break
            except Exception:
                continue
        
        if local_path and local_path.exists():
            lora_cache[cache_key] = str(local_path)
            return str(local_path)
        else:
            logger.error(f"❌ Could not find LoRA model in any expected location")
            return None
        
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

def apply_lora_composition(composition: Dict, pipeline=None):
    """Apply LoRA composition to the specified pipeline (or both if None)"""
    global pipe, controlnet_pipe
    
    try:
        # Clear GPU cache before loading LoRA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 Cleared GPU cache before LoRA loading")
        
        bucket = get_default_model_bucket()
        
        # Determine which pipelines to apply LoRA to
        pipelines = []
        if pipeline:
            pipelines = [pipeline]
        else:
            pipelines = [pipe, controlnet_pipe]
        
        for p in pipelines:
            # Reset adapters first
            try:
                p.unload_lora_weights()
                # Clear any existing adapter names
                if hasattr(p, '_adapter_weights'):
                    p._adapter_weights.clear()
                if hasattr(p, 'peft_config') and p.peft_config:
                    p.peft_config.clear()
            except Exception as e:
                logger.warning(f"⚠️ Could not fully unload LoRA weights: {e}")
            
            adapters = []
            weights = []
            
            # Load character LoRA if specified
            if 'character' in composition:
                char_config = composition['character']
                model_path = char_config['model_path']
                weight = char_config.get('weight', 1.0)
                
                lora_path = download_lora_model(model_path, bucket)
                if lora_path:
                    logger.info(f"🎭 Loading character LoRA with weight {weight} to {p.__class__.__name__}")
                    p.load_lora_weights(lora_path, adapter_name="character")
                    adapters.append("character")
                    weights.append(weight)
            
            # Load style LoRA if specified
            if 'style' in composition:
                style_config = composition['style']
                model_path = style_config['model_path']
                weight = style_config.get('weight', 0.7)
                
                lora_path = download_lora_model(model_path, bucket)
                if lora_path:
                    logger.info(f"🎨 Loading style LoRA with weight {weight} to {p.__class__.__name__}")
                    p.load_lora_weights(lora_path, adapter_name="style")
                    adapters.append("style")
                    weights.append(weight)
            
            # Apply adapters if any were loaded
            if adapters:
                p.set_adapters(adapters, adapter_weights=weights)
                logger.info(f"✅ Applied LoRA composition to {p.__class__.__name__}: {adapters} with weights {weights}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to apply LoRA composition: {str(e)}")
        return False

@app.route('/ping', methods=['GET'])
def ping():
    """SageMaker health check endpoint"""
    if pipe is None or controlnet_pipe is None:
        return flask.Response(status=503, response="Pipelines not loaded")
    
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
        if control_image_data:
            logger.warning(f"💥 control_type = {control_type}")
            logger.warning(f"✅ controlnet_pipe.controlnets = {controlnet_pipe.controlnets.keys()}")
            logger.warning(f"✅ controlnet_processors = {controlnet_processors.keys()}")
            
            if control_type in controlnet_pipe.controlnets and control_type in controlnet_processors:
                control_image = process_control_image(control_image_data, control_type)
            else:
                logger.error(f"❌ Control type '{control_type}' not available")
                control_image = None
            if control_image:
                # Swap ControlNet if needed
                if controlnet_pipe.controlnet != controlnet_pipe.controlnets[control_type]:
                    controlnet_pipe.controlnet = controlnet_pipe.controlnets[control_type]
                logger.info(f"✅ Using {control_type} ControlNet with conditioning scale {controlnet_conditioning_scale}")
        
        # Choose the appropriate pipeline and generate images
        if control_image:
            # Use ControlNet pipeline
            logger.info("🎨 Using ControlNet pipeline")
            generator = torch.Generator(device=controlnet_pipe.device).manual_seed(seed) if seed else None
            images = controlnet_pipe(
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
            # Use regular SDXL pipeline
            logger.info("🎨 Using regular SDXL pipeline")
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