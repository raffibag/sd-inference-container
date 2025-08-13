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
# SINGLE SOURCE OF TRUTH FOR DEVICE AND DTYPE + MULTI-GPU ALLOCATION
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type != "cuda":
    raise RuntimeError("CUDA not available. This service requires a GPU.")

DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32

# Autocast dtype - bf16 if available, else optimal for device
AMP_DTYPE = (
    torch.bfloat16 if (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported())
    else (torch.float16 if DEVICE.type == "cuda" else torch.bfloat16)
)

# Multi-GPU allocation strategy
GPU_COUNT = torch.cuda.device_count() if DEVICE.type == "cuda" else 0
GPU_ALLOCATION = {
    "base_pipeline": 0,      # Base SDXL on GPU 0
    "controlnet": 1,         # ControlNet on GPU 1 (if available)
    "depth_processor": 2,    # Depth processor on GPU 2 (if available)
    "openpose_processor": 3, # OpenPose on GPU 3 (if available)
    "canny_processor": 0,    # Canny shares with base (lightweight)
}

def get_optimal_gpu(component: str) -> int:
    """Get optimal GPU for component with fallback strategy"""
    if GPU_COUNT <= 1:
        return 0
    
    preferred = GPU_ALLOCATION.get(component, 0)
    
    # If preferred GPU exists, use it
    if preferred < GPU_COUNT:
        return preferred
    
    # Fallback strategy: distribute across available GPUs
    fallback_order = {
        "controlnet": [1, 2, 3, 0],
        "depth_processor": [2, 3, 1, 0], 
        "openpose_processor": [3, 2, 1, 0],
        "canny_processor": [0, 1, 2, 3],
        "base_pipeline": [0, 1, 2, 3]
    }
    
    for gpu_id in fallback_order.get(component, [0]):
        if gpu_id < GPU_COUNT:
            return gpu_id
    
    return 0  # Final fallback

logger.info(f"🔧 Device: {DEVICE}, Dtype: {DTYPE}, AMP: {AMP_DTYPE}")
if DEVICE.type == "cuda":
    logger.info(f"🔍 GPU Count: {GPU_COUNT}")
    for i in range(GPU_COUNT):
        logger.info(f"🔍 GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB)")
    logger.info(f"🔍 BF16 supported: {torch.cuda.is_bf16_supported()}")
    logger.info(f"🎯 GPU Allocation: Base={get_optimal_gpu('base_pipeline')}, CN={get_optimal_gpu('controlnet')}, Depth={get_optimal_gpu('depth_processor')}, OpenPose={get_optimal_gpu('openpose_processor')}")
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
        
        # Separate pipelines for memory efficiency
        self.pipe_base = None           # Base SDXL pipeline (no ControlNet)
        self.pipe_cn = None            # ControlNet pipeline (lazy loaded)
        self.loaded_controlnet = None  # Currently loaded ControlNet type
        
        # Processors and caching
        self.controlnet_processors = {}
        self.s3_client = None
        self.lora_cache = {}
        self.active_loras = {}
        self.adapter_paths = {}  # adapter_name -> local_path for reliable reapplication
        self.temp_dir = tempfile.mkdtemp(prefix="lora_cache_")
        
        self._initialize_base_pipeline()
        
    def _initialize_base_pipeline(self):
        """Initialize base SDXL pipeline (no ControlNet for memory efficiency)"""
        import boto3
        self.s3_client = boto3.client('s3')
        
        logger.info("🚀 Initializing base SDXL pipeline (memory-efficient)...")
        
        # Clear GPU cache before loading
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("🧹 Cleared GPU cache")
        
        # Create base SDXL pipeline (no ControlNet)
        from diffusers import StableDiffusionXLPipeline
        
        base_gpu = get_optimal_gpu("base_pipeline")
        base_device = torch.device(f"cuda:{base_gpu}")
        self.pipe_base = StableDiffusionXLPipeline.from_pretrained(
            self.base_model,
            torch_dtype=DTYPE,
            use_safetensors=True,
            add_watermarker=False
        ).to(base_device)
        
        # Keep VAE in fp32 to prevent black images
        if DEVICE.type == "cuda":
            self.pipe_base.vae.to(dtype=torch.float32)
            logger.info("✅ VAE set to fp32 to prevent black images")
        
        # Configure scheduler
        self.pipe_base.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe_base.scheduler.config,
            algorithm_type="dpmsolver++",
            solver_order=2,
            use_karras_sigmas=True
        )
        
        # Enable optimizations
        if DEVICE.type == "cuda":
            self.pipe_base.enable_attention_slicing()
            self.pipe_base.enable_vae_slicing()
            self.pipe_base.enable_vae_tiling()
            
            # Enable xformers if available
            try:
                self.pipe_base.enable_xformers_memory_efficient_attention()
                logger.info("✅ XFormers enabled")
            except:
                logger.info("⚠️ XFormers not available")
        
        logger.info("✅ Base pipeline initialized successfully (~7GB VRAM)")
    
    def _get_cn_pipeline(self, control_type: str):
        """Get ControlNet pipeline with aggressive VRAM management (max 1 CN in VRAM)"""
        
        # If already loaded with same ControlNet, reuse
        if self.pipe_cn and self.loaded_controlnet == control_type:
            return self.pipe_cn
        
        # Free previous CN pipeline aggressively
        if self.pipe_cn:
            try:
                del self.pipe_cn.controlnet
            except Exception:
                pass
            del self.pipe_cn
            self.pipe_cn = None
            torch.cuda.empty_cache()
            logger.info(f"🧹 Evicted previous ControlNet ({self.loaded_controlnet})")
        
        # ControlNet repository mapping
        controlnet_repos = {
            "canny": "diffusers/controlnet-canny-sdxl-1.0",
            "depth": "diffusers/controlnet-depth-sdxl-1.0", 
            "openpose": "xinsir/controlnet-openpose-sdxl-1.0",
        }
        
        if control_type not in controlnet_repos:
            raise ValueError(f"Unknown ControlNet type: {control_type}")
        
        repo_id = controlnet_repos[control_type]
        
        # Lazy-load requested ControlNet on optimal GPU
        cn_gpu = get_optimal_gpu("controlnet")
        cn_device = torch.device(f"cuda:{cn_gpu}")
        logger.info(f"🔄 Loading {control_type} ControlNet on demand (GPU {cn_gpu})...")
        
        cn = ControlNetModel.from_pretrained(
            repo_id,
            torch_dtype=DTYPE,
            use_safetensors=True
        ).to(cn_device)
        
        # Create ControlNet pipeline
        self.pipe_cn = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.base_model,
            controlnet=cn,
            torch_dtype=DTYPE,
            use_safetensors=True,
            add_watermarker=False
        ).to(cn_device)
        
        # Keep VAE in fp32 to prevent black images  
        if DEVICE.type == "cuda":
            self.pipe_cn.vae.to(dtype=torch.float32)
        
        # Apply same optimizations
        if DEVICE.type == "cuda":
            self.pipe_cn.enable_attention_slicing()
            self.pipe_cn.enable_vae_slicing()
            self.pipe_cn.enable_vae_tiling()
            
            try:
                self.pipe_cn.enable_xformers_memory_efficient_attention()
            except:
                pass
        
        # Configure scheduler to match base pipeline
        self.pipe_cn.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe_cn.scheduler.config,
            algorithm_type="dpmsolver++",
            solver_order=2,
            use_karras_sigmas=True
        )
        
        # Apply any active LoRAs to CN pipeline using stored paths
        if hasattr(self, 'active_loras') and self.active_loras:
            logger.info(f"🔄 Applying {len(self.active_loras)} LoRAs to ControlNet pipeline...")
            try:
                for adapter_name, weight in self.active_loras.items():
                    local_path = self.adapter_paths.get(adapter_name)
                    if local_path and os.path.exists(local_path):
                        self.pipe_cn.load_lora_weights(local_path, adapter_name=adapter_name)
                
                self.pipe_cn.set_adapters(
                    list(self.active_loras.keys()),
                    adapter_weights=list(self.active_loras.values())
                )
                logger.info("✅ LoRAs applied to ControlNet pipeline")
            except Exception as e:
                logger.warning(f"⚠️ Failed to apply some LoRAs to CN pipeline: {e}")
        
        self.loaded_controlnet = control_type
        logger.info(f"✅ {control_type} ControlNet loaded (~14GB VRAM)")
        
        return self.pipe_cn
    
    
    
    def _load_processor_on_demand(self, control_type: str):
        """Load a single processor on demand (USE OPTIMAL GPU FOR SPEED)"""
        try:
            if control_type == "canny":
                from controlnet_aux import CannyDetector
                self.controlnet_processors["canny"] = CannyDetector()
                logger.info("✅ Loaded Canny processor (CPU)")
            elif control_type == "openpose":
                from controlnet_aux import OpenposeDetector
                openpose_gpu = get_optimal_gpu("openpose_processor")
                self.controlnet_processors["openpose"] = OpenposeDetector.from_pretrained(
                    "lllyasviel/Annotators",
                    device=f"cuda:{openpose_gpu}"
                )
                logger.info(f"✅ Loaded OpenPose processor (GPU {openpose_gpu})")
            elif control_type == "depth":
                from transformers import pipeline
                depth_gpu = get_optimal_gpu("depth_processor")
                self.controlnet_processors["depth"] = pipeline(
                    "depth-estimation",
                    model="Intel/dpt-large",
                    device=depth_gpu if DEVICE.type == "cuda" else -1
                )
                logger.info(f"✅ Loaded Depth processor (GPU {depth_gpu})")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load {control_type} processor: {e}")
    
    
    def _resize_control_image(self, img: Union[Image.Image, np.ndarray], width: int, height: int):
        """Resize control image to match requested dimensions"""
        if isinstance(img, np.ndarray):
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        else:
            return img.resize((width, height), Image.BICUBIC)
    
    def _ensure_pil_rgb(self, img: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Normalize control image to RGB PIL format"""
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.ndim == 3 and img.shape[2] == 3:
                pass
            else:
                raise ValueError(f"Unexpected control image shape: {img.shape}")
            img = Image.fromarray(img)
        elif isinstance(img, Image.Image):
            if img.mode != "RGB":
                img = img.convert("RGB")
        else:
            raise TypeError(f"Unsupported control image type: {type(img)}")
        return img
    
    def _batch_control(self, img: Union[Image.Image, np.ndarray], num: int) -> List[Image.Image]:
        """Convert control image to proper batch format for Diffusers"""
        pil = self._ensure_pil_rgb(img)
        return [pil] * num  # broadcast to batch
    
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
            
            # Load processor on demand if not already loaded
            if control_type.value not in self.controlnet_processors:
                self._load_processor_on_demand(control_type.value)
            
            processor = self.controlnet_processors.get(control_type.value)
            if not processor:
                logger.warning(f"No processor for {control_type.value}, using raw image")
                return control_image
            
            # Process based on type
            if control_type == ControlNetType.canny:
                # Canny returns numpy array - convert to PIL RGB
                opencv_image = cv2.cvtColor(np.array(control_image), cv2.COLOR_RGB2BGR)
                processed = processor(opencv_image)  # HxW uint8
                # Ensure correct size
                processed = self._resize_control_image(processed, width, height)
                # Convert to 3-channel if needed, then PIL RGB
                if processed.ndim == 2:
                    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
                return Image.fromarray(processed).convert("RGB")
                
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
                
                # Resize and return as PIL RGB image
                depth_pil = Image.fromarray(depth_3channel)
                depth_resized = self._resize_control_image(depth_pil, width, height)
                return depth_resized.convert("RGB")
                
            elif control_type == ControlNetType.openpose:
                # OpenPose returns detected pose - ensure RGB PIL
                processed = processor(control_image)
                # Ensure correct size
                processed = self._resize_control_image(processed, width, height)
                # Ensure RGB PIL format
                if isinstance(processed, np.ndarray):
                    processed = Image.fromarray(processed)
                return processed.convert("RGB")
                
            else:
                return control_image
                
        except Exception as e:
            logger.error(f"Error processing control image: {e}")
            return None
    
    def _verify_device_consistency(self, pipe):
        """Verify all components are on the same device with same dtype"""
        try:
            # Check UNet
            unet_device = next(pipe.unet.parameters()).device
            unet_dtype = next(pipe.unet.parameters()).dtype
            logger.info(f"🔍 UNet: {unet_device}, {unet_dtype}")
            
            # Check ControlNet if present
            if hasattr(pipe, 'controlnet') and pipe.controlnet is not None:
                cn_device = next(pipe.controlnet.parameters()).device
                cn_dtype = next(pipe.controlnet.parameters()).dtype
                logger.info(f"🔍 ControlNet: {cn_device}, {cn_dtype}")
                
                # Assert consistency
                assert unet_device == cn_device, f"Device mismatch: UNet {unet_device} vs ControlNet {cn_device}"
                assert unet_dtype == cn_dtype, f"Dtype mismatch: UNet {unet_dtype} vs ControlNet {cn_dtype}"
            
            # Check VAE
            vae_device = next(pipe.vae.parameters()).device
            vae_dtype = next(pipe.vae.parameters()).dtype
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
            
            # Load LoRA on base pipeline
            try:
                # First try as single file
                self.pipe_base.load_lora_weights(local_path, adapter_name=adapter_name)
            except Exception as e1:
                # Try as folder with adapter_config.json
                try:
                    folder = os.path.dirname(local_path) if os.path.isfile(local_path) else local_path
                    self.pipe_base.load_lora_weights(folder, adapter_name=adapter_name)
                except Exception as e2:
                    logger.error(f"Failed both loading methods: file={e1}, folder={e2}")
                    raise e2
            
            # Track this LoRA and store its path
            self.active_loras[adapter_name] = weight
            self.adapter_paths[adapter_name] = local_path
            
            # Apply to base pipeline
            self.pipe_base.set_adapters(
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
            # Clear from base pipeline
            try:
                self.pipe_base.unload_lora_weights()
                self.pipe_base.set_adapters([], adapter_weights=[])
            except Exception as e:
                logger.debug(f"No LoRA weights to unload from base: {e}")
            
            # Clear from CN pipeline if loaded
            if self.pipe_cn:
                try:
                    self.pipe_cn.unload_lora_weights()
                    self.pipe_cn.set_adapters([], adapter_weights=[])
                except Exception as e:
                    logger.debug(f"No LoRA weights to unload from CN: {e}")
            
            self.active_loras = {}
            self.adapter_paths = {}
            logger.info("Cleared all LoRAs from both pipelines")
    
    def _set_scheduler_for_pipeline(self, pipeline, scheduler_type: SchedulerType):
        """Configure scheduler for a specific pipeline"""
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
            pipeline.scheduler = scheduler_map[scheduler_type](pipeline.scheduler.config)
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
        
        # Set scheduler on base pipeline 
        self._set_scheduler_for_pipeline(self.pipe_base, scheduler)
        
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
            # Process control image (returns PIL or numpy) with correct size
            control_image = self.process_control_image(control_image_data, control_type, width, height)
            if control_image is not None:
                logger.info(f"✅ Using {control_type.value} ControlNet")
            else:
                logger.warning("Control image processing failed, falling back to base generation")
                control_type = ControlNetType.none
        
        # IMPORTANT: Generator must be on the same device
        if control_type == ControlNetType.none:
            pipe = self.pipe_base
        else:
            pipe = self._get_cn_pipeline(control_type.value)
        self._set_scheduler_for_pipeline(pipe, scheduler)

        pipe_device = next(pipe.unet.parameters()).device
        generator = torch.Generator(device=pipe_device).manual_seed(seed)
        
        logger.info(f"🎨 Generating {num_images} image(s) at {width}x{height}")
        logger.info(f"🎨 ControlNet: {control_type.value}, LoRAs: {loras_loaded}, Seed: {seed}")
        
        # Generate with appropriate pipeline
        try:
            # Use autocast and inference mode for performance
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                if control_type == ControlNetType.none:
                    images = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        num_images_per_prompt=num_images,
                        generator=generator,
                    ).images
                else:
                    control_batch = self._batch_control(control_image, num_images)
                    images = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=control_batch,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        width=width,
                        height=height,
                        num_images_per_prompt=num_images,
                        generator=generator,
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
            try:
                if control_type == ControlNetType.none:
                    self._verify_device_consistency(self.pipe_base)
                else:
                    self._verify_device_consistency(self.pipe_cn)
            except:
                pass  # Don't let verification errors mask the original error
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
    
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "loaded_controlnet": composer.loaded_controlnet if composer.loaded_controlnet else "none",
        "memory_mode": "lazy_loading",
        "pipelines": {
            "base": "loaded" if composer.pipe_base else "none",
            "controlnet": "loaded" if composer.pipe_cn else "none"
        }
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