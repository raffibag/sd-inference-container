#!/usr/bin/env python3
"""
Production-ready SDXL Multi-LoRA Inference with ControlNet
Implements all device/dtype consistency fixes and production robustness
"""

import os
import sys
import json
import logging
import torch
import numpy as np
import cv2
import base64
import random
import tempfile
from io import BytesIO
from PIL import Image
from typing import Dict, List, Optional, Union, Tuple
from enum import Enum
from pathlib import Path

from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler
)

# FastAPI imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Vibez Multi-LoRA Inference API")

# ==============================================================================
# SINGLE SOURCE OF TRUTH FOR DEVICE AND DTYPE
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32  # Use float32 consistently to avoid black image issues

logger.info(f"🔧 Device: {DEVICE}, Dtype: {DTYPE}")
if DEVICE.type == "cuda":
    logger.info(f"🔍 CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"🔍 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    # Enable cudnn benchmark for performance
    torch.backends.cudnn.benchmark = True

# ==============================================================================
# ENUMS AND MODELS
# ==============================================================================

class SchedulerType(str, Enum):
    dpmpp_2m_karras = "dpmpp_2m_karras"
    dpmpp_2m = "dpmpp_2m"
    euler = "euler"
    euler_a = "euler_a"
    ddim = "ddim"
    pndm = "pndm"
    unipc = "unipc"

class ControlNetType(str, Enum):
    canny = "canny"
    depth = "depth"
    openpose = "openpose"
    none = "none"

class InferenceRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    composition: Optional[Dict] = Field(default_factory=dict)
    scheduler: Optional[SchedulerType] = SchedulerType.dpmpp_2m_karras
    control_type: Optional[ControlNetType] = ControlNetType.none
    control_image: Optional[str] = None  # Base64 encoded
    controlnet_conditioning_scale: Optional[float] = 1.0
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    num_images: Optional[int] = 1
    seed: Optional[int] = None

# ==============================================================================
# CONTROLNET LOADER WITH CONSISTENT DEVICE/DTYPE
# ==============================================================================

def load_controlnet(repo_id: str, name: str = None) -> ControlNetModel:
    """Load a ControlNet with consistent device and dtype"""
    try:
        logger.info(f"Loading {name or repo_id} ControlNet...")
        cn = ControlNetModel.from_pretrained(
            repo_id,
            torch_dtype=DTYPE,
            use_safetensors=True
        )
        cn = cn.to(DEVICE)
        logger.info(f"✅ Loaded {name or repo_id} on {DEVICE} with {DTYPE}")
        return cn
    except Exception as e:
        logger.error(f"❌ Failed to load {name or repo_id}: {e}")
        raise

# ==============================================================================
# MULTI-LORA COMPOSER CLASS
# ==============================================================================

class MultiLoRAComposer:
    def __init__(self, base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        self.base_model = base_model
        self.pipeline = None
        self.controlnets = {}
        self.controlnet_processors = {}
        self.s3_client = None
        self.lora_cache = {}
        self.active_loras = {}
        self.temp_dir = tempfile.mkdtemp(prefix="lora_cache_")
        
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Initialize SDXL pipeline with ControlNet support"""
        import boto3
        self.s3_client = boto3.client('s3')
        
        logger.info("🚀 Initializing SDXL ControlNet pipeline...")
        
        # Clear GPU cache before loading
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("🧹 Cleared GPU cache")
        
        # Load initial ControlNet (canny) with consistent device/dtype
        initial_controlnet = load_controlnet(
            "diffusers/controlnet-canny-sdxl-1.0-small",
            name="canny"
        )
        
        # Create pipeline with consistent dtype
        self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.base_model,
            controlnet=initial_controlnet,
            torch_dtype=DTYPE,
            use_safetensors=True,
            add_watermarker=False
        )
        
        # Move entire pipeline to device
        self.pipeline = self.pipeline.to(DEVICE)
        
        # Cache the initial controlnet
        self.controlnets["canny"] = initial_controlnet
        
        # Load additional ControlNets
        self._load_additional_controlnets()
        
        # Configure scheduler
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config,
            algorithm_type="dpmsolver++",
            solver_order=2,
            use_karras_sigmas=True
        )
        
        # Enable optimizations
        if DEVICE.type == "cuda":
            self.pipeline.enable_attention_slicing()
            # Enable VAE optimizations for large images
            self.pipeline.enable_vae_slicing()
            self.pipeline.enable_vae_tiling()
            
            # Enable xformers if available
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("✅ XFormers enabled")
            except:
                logger.info("⚠️ XFormers not available")
        
        # Verify device consistency
        self._verify_device_consistency()
        
        logger.info("✅ Pipeline initialized successfully")
    
    def _load_additional_controlnets(self):
        """Load additional ControlNet models"""
        controlnet_configs = {
            "depth": "diffusers/controlnet-depth-sdxl-1.0",
            "openpose": "thibaud/controlnet-openpose-sdxl-1.0"
        }
        
        for name, repo_id in controlnet_configs.items():
            try:
                cn = load_controlnet(repo_id, name)
                self.controlnets[name] = cn
            except Exception as e:
                logger.warning(f"⚠️ Could not load {name}: {e}")
        
        # Load processors
        self._load_processors()
    
    def _load_processors(self):
        """Load ControlNet preprocessors"""
        try:
            from controlnet_aux import CannyDetector, OpenposeDetector
            from transformers import pipeline
            
            self.controlnet_processors["canny"] = CannyDetector()
            self.controlnet_processors["openpose"] = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            
            # Load depth processor with device specification
            self.controlnet_processors["depth"] = pipeline(
                "depth-estimation",
                model="Intel/dpt-large",
                device=0 if DEVICE.type == "cuda" else -1
            )
            
            logger.info("✅ Loaded ControlNet processors")
        except Exception as e:
            logger.warning(f"⚠️ Some processors not available: {e}")
    
    def switch_controlnet(self, control_type: ControlNetType):
        """Switch to a different ControlNet"""
        if control_type == ControlNetType.none:
            return True
            
        if control_type.value not in self.controlnets:
            logger.warning(f"⚠️ ControlNet {control_type.value} not available")
            return False
        
        new_cn = self.controlnets[control_type.value]
        
        # Ensure the new ControlNet is on the same device/dtype
        new_cn = new_cn.to(device=DEVICE, dtype=DTYPE)
        
        # Assign to pipeline
        self.pipeline.controlnet = new_cn
        
        logger.info(f"🔄 Switched to {control_type.value} ControlNet")
        
        # Verify consistency after switch (skip if using offload)
        if not hasattr(self.pipeline, '_offload_hooks'):
            self._verify_device_consistency()
        
        return True
    
    def _resize_control_image(self, img: Union[Image.Image, np.ndarray], width: int, height: int):
        """Resize control image to match requested dimensions"""
        if isinstance(img, np.ndarray):
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        else:
            return img.resize((width, height), Image.BICUBIC)
    
    def process_control_image(self, image_data: str, control_type: ControlNetType, width: int, height: int) -> Union[Image.Image, np.ndarray, None]:
        """
        Process control image and return PIL or numpy array
        Let diffusers handle the conversion to tensor
        """
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            control_image = Image.open(BytesIO(image_bytes)).convert('RGB')
            
            # Resize to match requested dimensions first
            control_image = control_image.resize((width, height), Image.BICUBIC)
            
            processor = self.controlnet_processors.get(control_type.value)
            if not processor:
                logger.warning(f"No processor for {control_type.value}, using raw image")
                return control_image
            
            # Process based on type
            if control_type == ControlNetType.canny:
                # Canny returns numpy array
                opencv_image = cv2.cvtColor(np.array(control_image), cv2.COLOR_RGB2BGR)
                processed = processor(opencv_image)
                # Ensure correct size
                processed = self._resize_control_image(processed, width, height)
                return processed  # Return numpy array
                
            elif control_type == ControlNetType.depth:
                # Depth estimation returns PIL
                depth_result = processor(control_image)
                depth_image = depth_result['depth']
                
                # Convert to normalized array with edge case handling
                depth_array = np.array(depth_image)
                denom = depth_array.max() - depth_array.min()
                
                if denom < 1e-8:
                    # Flat depth map - use neutral gray
                    depth_normalized = np.ones_like(depth_array, dtype=np.uint8) * 127
                else:
                    depth_normalized = ((depth_array - depth_array.min()) / denom * 255).astype(np.uint8)
                
                # Convert to 3-channel
                if len(depth_normalized.shape) == 2:
                    depth_3channel = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2RGB)
                else:
                    depth_3channel = depth_normalized
                
                # Resize and return as PIL image
                depth_pil = Image.fromarray(depth_3channel)
                return self._resize_control_image(depth_pil, width, height)
                
            elif control_type == ControlNetType.openpose:
                # OpenPose returns detected pose
                processed = processor(control_image)
                # Ensure correct size
                processed = self._resize_control_image(processed, width, height)
                return processed
                
            else:
                return control_image
                
        except Exception as e:
            logger.error(f"Error processing control image: {e}")
            return None
    
    def _verify_device_consistency(self):
        """Verify all components are on the same device with same dtype"""
        try:
            # Check UNet
            unet_device = next(self.pipeline.unet.parameters()).device
            unet_dtype = next(self.pipeline.unet.parameters()).dtype
            logger.info(f"🔍 UNet: {unet_device}, {unet_dtype}")
            
            # Check ControlNet
            if hasattr(self.pipeline, 'controlnet'):
                cn_device = next(self.pipeline.controlnet.parameters()).device
                cn_dtype = next(self.pipeline.controlnet.parameters()).dtype
                logger.info(f"🔍 ControlNet: {cn_device}, {cn_dtype}")
                
                # Assert consistency
                assert unet_device == cn_device, f"Device mismatch: UNet {unet_device} vs ControlNet {cn_device}"
                assert unet_dtype == cn_dtype, f"Dtype mismatch: UNet {unet_dtype} vs ControlNet {cn_dtype}"
            
            # Check VAE
            vae_device = next(self.pipeline.vae.parameters()).device
            vae_dtype = next(self.pipeline.vae.parameters()).dtype
            logger.info(f"🔍 VAE: {vae_device}, {vae_dtype}")
            
            logger.info("✅ Device consistency verified")
            
        except Exception as e:
            logger.error(f"❌ Device consistency check failed: {e}")
            raise
    
    def load_lora(self, model_path: str, weight: float = 1.0):
        """Load a LoRA model from S3 or local path with fallback support"""
        try:
            # Initialize active_loras tracking if needed
            if not hasattr(self, 'active_loras'):
                self.active_loras = {}
            
            # Handle S3 paths
            local_path = model_path
            if not os.path.exists(model_path):
                local_path = self._download_lora_from_s3(model_path)
            
            # Generate adapter name from path
            adapter_name = model_path.replace('/', '_').replace(':', '_')
            
            # Try loading the LoRA weights
            try:
                # First try as single file
                self.pipeline.load_lora_weights(local_path, adapter_name=adapter_name)
            except Exception as e1:
                # Try as folder with adapter_config.json
                try:
                    folder = os.path.dirname(local_path) if os.path.isfile(local_path) else local_path
                    self.pipeline.load_lora_weights(folder, adapter_name=adapter_name)
                except Exception as e2:
                    logger.error(f"Failed both loading methods: file={e1}, folder={e2}")
                    raise e2
            
            # Track this LoRA
            self.active_loras[adapter_name] = weight
            
            # Apply all active LoRAs
            self.pipeline.set_adapters(
                list(self.active_loras.keys()),
                adapter_weights=list(self.active_loras.values())
            )
            
            logger.info(f"✅ Loaded LoRA {adapter_name} with weight {weight}")
            return True
            
        except Exception as e:
            logger.error(f"❌ LoRA load failed for {model_path}: {e}")
            return False
    
    def _download_lora_from_s3(self, model_path: str) -> str:
        """Download LoRA from S3 to temporary location"""
        try:
            # Parse S3 path
            if model_path.startswith('s3://'):
                path = model_path[5:]
                parts = path.split('/', 1)
                bucket = parts[0]
                key = parts[1] if len(parts) > 1 else ''
            else:
                # Assume model registry format
                bucket = os.environ.get('MODEL_BUCKET', 'vibez-model-registry-796245059390-us-west-2')
                # Try multiple possible keys
                possible_keys = [
                    f"models/{model_path}/pytorch_lora_weights.safetensors",
                    f"models/{model_path}/model.safetensors",
                    f"models/{model_path}/adapter_model.bin"
                ]
                key = None
                
                # Find which key exists
                for test_key in possible_keys:
                    try:
                        self.s3_client.head_object(Bucket=bucket, Key=test_key)
                        key = test_key
                        break
                    except:
                        continue
                
                if not key:
                    raise FileNotFoundError(f"No LoRA file found for {model_path}")
            
            # Create temp file
            safe_name = model_path.replace('/', '_').replace(':', '')
            local_path = os.path.join(self.temp_dir, f"{safe_name}.safetensors")
            
            # Download from S3
            logger.info(f"Downloading LoRA from s3://{bucket}/{key}")
            self.s3_client.download_file(bucket, key, local_path)
            
            # Also try to download adapter_config.json if it exists
            config_key = key.rsplit('/', 1)[0] + '/adapter_config.json' if '/' in key else 'adapter_config.json'
            config_path = os.path.join(self.temp_dir, f"{safe_name}_config.json")
            try:
                self.s3_client.download_file(bucket, config_key, config_path)
                logger.info(f"Also downloaded adapter_config.json")
            except:
                pass  # Config is optional
            
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download LoRA from S3: {e}")
            raise
    
    def clear_loras(self):
        """Clear all loaded LoRAs - safe to call even if no LoRAs loaded"""
        if hasattr(self, 'active_loras') and self.active_loras:
            try:
                # Unload LoRA weights
                self.pipeline.unload_lora_weights()
            except Exception as e:
                logger.debug(f"No LoRA weights to unload: {e}")
            
            try:
                # Clear adapter settings
                self.pipeline.set_adapters([], adapter_weights=[])
            except Exception as e:
                logger.debug(f"No adapters to clear: {e}")
            
            self.active_loras = {}
            logger.info("Cleared all LoRAs")
    
    def set_scheduler(self, scheduler_type: SchedulerType):
        """Configure scheduler"""
        scheduler_map = {
            SchedulerType.dpmpp_2m_karras: lambda config: DPMSolverMultistepScheduler.from_config(
                config, algorithm_type="dpmsolver++", solver_order=2, use_karras_sigmas=True
            ),
            SchedulerType.dpmpp_2m: lambda config: DPMSolverMultistepScheduler.from_config(
                config, algorithm_type="dpmsolver++", solver_order=2, use_karras_sigmas=False
            ),
            SchedulerType.euler: lambda config: EulerDiscreteScheduler.from_config(config),
            SchedulerType.euler_a: lambda config: EulerAncestralDiscreteScheduler.from_config(config),
            SchedulerType.ddim: lambda config: DDIMScheduler.from_config(config),
            SchedulerType.pndm: lambda config: PNDMScheduler.from_config(config),
            SchedulerType.unipc: lambda config: UniPCMultistepScheduler.from_config(config),
        }
        
        if scheduler_type in scheduler_map:
            self.pipeline.scheduler = scheduler_map[scheduler_type](self.pipeline.scheduler.config)
            logger.info(f"✅ Scheduler set to {scheduler_type.value}")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        composition: Dict = None,
        scheduler: SchedulerType = SchedulerType.dpmpp_2m_karras,
        control_type: ControlNetType = ControlNetType.none,
        control_image_data: Optional[str] = None,
        controlnet_conditioning_scale: float = 1.0,
        steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        seed: Optional[int] = None
    ) -> Dict:
        """Generate images with optional ControlNet and LoRA composition"""
        
        # Generate or use provided seed
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            logger.info(f"Generated random seed: {seed}")
        
        # Set scheduler
        self.set_scheduler(scheduler)
        
        # Clear any previous LoRAs (important for stateless inference)
        self.clear_loras()
        
        # Track which features are active
        loras_loaded = False
        
        # Load LoRAs if specified (optional - works without any LoRAs)
        if composition:
            for lora_type, config in composition.items():
                if isinstance(config, dict) and 'model_path' in config:
                    if self.load_lora(config['model_path'], config.get('weight', 1.0)):
                        loras_loaded = True
            # No error if no LoRAs load - that's valid
        
        # Process ControlNet if specified
        control_image = None
        if control_type != ControlNetType.none and control_image_data:
            # Switch to appropriate ControlNet
            if not self.switch_controlnet(control_type):
                logger.warning("ControlNet switch failed, using regular generation")
                control_type = ControlNetType.none
            else:
                # Process control image (returns PIL or numpy) with correct size
                control_image = self.process_control_image(control_image_data, control_type, width, height)
                if control_image is not None:
                    logger.info(f"✅ Using {control_type.value} ControlNet")
                else:
                    logger.warning("Control image processing failed")
                    control_type = ControlNetType.none
        
        # IMPORTANT: Generator must be on the same device
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        
        logger.info(f"🎨 Generating {num_images} image(s) at {width}x{height}")
        logger.info(f"🎨 ControlNet: {control_type.value}, LoRAs: {loras_loaded}, Seed: {seed}")
        
        # Generate with or without ControlNet
        try:
            # Use autocast and inference mode for performance
            with torch.inference_mode(), torch.autocast(
                device_type="cuda" if DEVICE.type == "cuda" else "cpu",
                dtype=DTYPE
            ):
                if control_image is not None:
                    # ControlNet generation - pass PIL/numpy, let diffusers handle conversion
                    images = self.pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=control_image,  # PIL or numpy - diffusers will handle
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        width=width,
                        height=height,
                        num_images_per_prompt=num_images,
                        generator=generator
                    ).images
                else:
                    # Regular gen with ControlNet effectively disabled
                    # Use neutral gray instead of white to avoid bias
                    blank = Image.new("RGB", (width, height), (127, 127, 127))
                    images = self.pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=blank,  # Required by ControlNet pipeline
                        controlnet_conditioning_scale=0.0,  # Disable ControlNet
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        num_images_per_prompt=num_images,
                        generator=generator
                    ).images
            
            # Convert images to base64
            result_images = []
            for img in images:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                result_images.append(img_base64)
            
            logger.info(f"✅ Generated {len(result_images)} images")
            
            # Return with metadata for reproducibility
            return {
                "images": result_images,
                "metadata": {
                    "seed": seed,
                    "control_type": control_type.value,
                    "loras_active": loras_loaded,
                    "scheduler": scheduler.value,
                    "steps": steps,
                    "guidance_scale": guidance_scale,
                    "width": width,
                    "height": height
                }
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Log device info for debugging
            self._verify_device_consistency()
            raise
    
    def cleanup(self):
        """Clean up temporary resources"""
        try:
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")

# ==============================================================================
# GLOBAL INSTANCE
# ==============================================================================

composer = None

# ==============================================================================
# FASTAPI ENDPOINTS
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the composer on startup"""
    global composer
    composer = MultiLoRAComposer()
    logger.info("🚀 API ready for inference")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    global composer
    if composer:
        composer.cleanup()
        composer = None
    logger.info("👋 API shutdown complete")

@app.get("/ping")
async def health_check():
    """Health check endpoint with enriched info"""
    if composer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get current ControlNet
    current_cn = "unknown"
    if hasattr(composer.pipeline, 'controlnet'):
        for name, cn in composer.controlnets.items():
            if composer.pipeline.controlnet == cn:
                current_cn = name
                break
    
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "current_controlnet": current_cn,
        "available_controlnets": list(composer.controlnets.keys())
    }

@app.post("/invocations")
async def generate_image(request: InferenceRequest):
    """Main inference endpoint"""
    if composer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = composer.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            composition=request.composition,
            scheduler=request.scheduler,
            control_type=request.control_type,
            control_image_data=request.control_image,
            controlnet_conditioning_scale=request.controlnet_conditioning_scale,
            steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            num_images=request.num_images,
            seed=request.seed
        )
        return result
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)