"""
Multi-LoRA Composition Inference for Character Generation
Refactored for FastAPI with async handlers and memory optimizations
"""

import os
import json
import time
import torch
import boto3
import logging
import cv2
import numpy as np
from typing import Dict, List, Optional
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from enum import Enum
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
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
composer = None

class SchedulerType(str, Enum):
    dpmpp_2m_karras = "dpmpp_2m_karras"  # Default - DPM++ 2M Karras (recommended)
    dpmpp_2m = "dpmpp_2m"                # DPM++ 2M without Karras
    euler = "euler"                      # Euler discrete
    euler_a = "euler_a"                  # Euler ancestral (adds noise)
    ddim = "ddim"                        # DDIM (deterministic)
    pndm = "pndm"                        # PNDM (pseudo numerical)
    unipc = "unipc"                      # UniPC (fast, high quality)

class ControlNetType(str, Enum):
    canny = "canny"                      # Edge detection control
    depth = "depth"                      # Depth map control
    openpose = "openpose"                # Human pose control
    none = "none"                        # No ControlNet (regular SDXL)

class GenerateRequest(BaseModel):
    composition: Dict
    character_description: Optional[str] = ""
    prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    use_ai_prompts: Optional[bool] = True
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    num_images: Optional[int] = 1
    seed: Optional[int] = None
    scheduler: Optional[SchedulerType] = SchedulerType.dpmpp_2m_karras
    # ControlNet parameters
    control_type: Optional[ControlNetType] = ControlNetType.none
    control_image: Optional[str] = None  # Base64 encoded image
    controlnet_conditioning_scale: Optional[float] = 1.0

class MultiLoRAComposer:
    def __init__(self, model_bucket: str, base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_bucket = model_bucket
        self.base_model = base_model
        self.s3 = boto3.client('s3')
        self.bedrock = boto3.client('bedrock-runtime')
        self.lora_cache = {}
        
        # Registry caching (5 minute cache)
        self.registry_cache = {}
        self.registry_loaded_at = 0
        
        # ControlNet models cache
        self.controlnets = {}
        self.controlnet_processors = {}

        logger.info("Loading SDXL ControlNet pipeline...")
        
        # Load initial ControlNet (canny) for the pipeline
        initial_controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0-small",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Create ControlNet pipeline (handles both regular SDXL and ControlNet)
        self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model,
            controlnet=initial_controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if self.device == "cuda" else None
        ).to(self.device)
        
        # Cache the initial controlnet
        self.controlnets["canny"] = initial_controlnet
        
        # Load other ControlNets
        self._load_controlnets()

        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config,
            algorithm_type="dpmsolver++",
            solver_order=2,
            use_karras_sigmas=True
        )

        if self.device == "cuda":
            # Enable memory optimizations
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("✅ xformers enabled")
            except Exception as e:
                logger.warning(f"⚠️ xformers failed: {e}")
            self.pipeline.enable_model_cpu_offload()
            logger.info("✅ CPU offload enabled")

    def _load_controlnets(self):
        """Load additional ControlNet models and processors"""
        try:
            from controlnet_aux import CannyDetector, OpenposeDetector
            from transformers import pipeline
            
            # ControlNet models
            controlnet_configs = {
                "depth": "diffusers/controlnet-depth-sdxl-1.0",
                "openpose": "thibaud/controlnet-openpose-sdxl-1.0"
            }
            
            # Load ControlNet models
            for name, model_id in controlnet_configs.items():
                try:
                    logger.info(f"Loading {name} ControlNet...")
                    controlnet = ControlNetModel.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                    )
                    self.controlnets[name] = controlnet
                    logger.info(f"✅ Loaded {name} ControlNet")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {name} ControlNet: {e}")
            
            # Load processors
            try:
                self.controlnet_processors["canny"] = CannyDetector()
                self.controlnet_processors["openpose"] = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
                self.controlnet_processors["depth"] = pipeline("depth-estimation", model="Intel/dpt-large")
                logger.info("✅ Loaded ControlNet processors")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load some processors: {e}")
                
        except ImportError as e:
            logger.warning(f"⚠️ ControlNet dependencies not available: {e}")

    def set_scheduler(self, scheduler_type: SchedulerType):
        """Configure scheduler based on type"""
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
        else:
            logger.warning(f"⚠️ Unknown scheduler {scheduler_type}, keeping default")

    def load_model_registry(self) -> Dict:
        # Return cached registry if within 5 minutes
        if time.time() - self.registry_loaded_at < 300:
            logger.debug("Using cached model registry")
            return self.registry_cache
            
        try:
            logger.info("Loading fresh model registry from S3")
            obj = self.s3.get_object(Bucket=self.model_bucket, Key="registry/models.json")
            registry = json.loads(obj["Body"].read())
            
            # Update cache
            self.registry_loaded_at = time.time()
            self.registry_cache = registry
            
            return registry
        except Exception as e:
            logger.warning(f"Model registry load failed: {e}")
            # Return cached registry if available, otherwise empty
            return self.registry_cache if self.registry_cache else {}

    def process_control_image(self, image_data: str, control_type: ControlNetType) -> Optional[Image.Image]:
        """Process control image using specified ControlNet processor"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            control_image = Image.open(BytesIO(image_bytes)).convert('RGB')
            
            processor = self.controlnet_processors.get(control_type.value)
            if not processor:
                logger.error(f"❌ Unknown control type: {control_type.value}")
                return None
            
            logger.info(f"🔍 Processing control image with {control_type.value}")
            
            if control_type == ControlNetType.canny:
                # Convert PIL to OpenCV format for Canny
                opencv_image = cv2.cvtColor(np.array(control_image), cv2.COLOR_RGB2BGR)
                processed = processor(opencv_image)
                return Image.fromarray(processed)
            elif control_type == ControlNetType.depth:
                # Use HuggingFace depth estimation pipeline
                depth_result = processor(control_image)
                depth_image = depth_result['depth']
                # Convert to PIL Image
                if hasattr(depth_image, 'convert'):
                    return depth_image.convert('RGB')
                else:
                    return Image.fromarray((depth_image * 255).astype(np.uint8))
            else:
                # Use processor directly (openpose)
                processed = processor(control_image)
                return processed
                
        except Exception as e:
            logger.error(f"❌ Failed to process control image: {str(e)}")
            return None

    def download_lora(self, model_path: str) -> str:
        if model_path in self.lora_cache:
            return self.lora_cache[model_path]

        local_path = f"/tmp/lora_models/{model_path}"
        os.makedirs(local_path, exist_ok=True)

        try:
            self.s3.download_file(self.model_bucket, f"models/{model_path}/pytorch_lora_weights.safetensors", f"{local_path}/pytorch_lora_weights.safetensors")
            self.s3.download_file(self.model_bucket, f"models/{model_path}/adapter_config.json", f"{local_path}/adapter_config.json")
            self.lora_cache[model_path] = local_path
            return local_path
        except Exception as e:
            logger.error(f"Failed to download LoRA model: {e}")
            return None

    def generate_intelligent_prompt(self, composition: Dict, character_description: str) -> Dict[str, str]:
        try:
            registry = self.load_model_registry()
            context = []
            trigger_words = []

            for comp_type, config in composition.items():
                if isinstance(config, dict) and 'model_path' in config:
                    path = config['model_path']
                    weight = config.get('weight', 1.0)

                    for project, pdata in registry.get("projects", {}).items():
                        for mname, minfo in pdata.get("models", {}).items():
                            if minfo.get("path") == path:
                                trig = minfo.get("trigger_word", "")
                                if trig:
                                    trigger_words.append(trig)
                                context.append(f"{comp_type}: {trig} (weight: {weight})")
                                meta = minfo.get("metadata", {})
                                if meta:
                                    context.append(f"  - Style: {meta.get('style', 'N/A')}")
                                    if "dominant_colors" in meta:
                                        context.append(f"  - Colors: {', '.join(meta['dominant_colors'])}")

            prompt_template = f"""You are an expert Stable Diffusion XL prompt engineer. Create optimized prompts.

Character Description: {character_description or 'Epic fantasy character'}

Composition Details:
{chr(10).join(context)}

Required trigger words: {', '.join(trigger_words)}

Return JSON:
{{"positive_prompt": "...", "negative_prompt": "..."}}"""

            response = self.bedrock.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt_template}]
                })
            )

            raw = json.loads(response["body"].read())
            text = raw["content"][0]["text"]
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group()) if match else {}
        except Exception as e:
            logger.error(f"Claude prompt gen failed: {e}")
            return {
                "positive_prompt": f"high quality, {', '.join(trigger_words)}, cinematic lighting, photorealistic",
                "negative_prompt": "blurry, deformed, lowres, extra limbs, bad anatomy"
            }

    async def generate_images(
        self,
        composition: Dict,
        prompt: str,
        negative_prompt: str,
        steps: int,
        scale: float,
        width: int,
        height: int,
        num_images: int,
        seed: Optional[int] = None,
        scheduler: SchedulerType = SchedulerType.dpmpp_2m_karras,
        control_type: ControlNetType = ControlNetType.none,
        control_image_data: Optional[str] = None,
        controlnet_conditioning_scale: float = 1.0
    ) -> List[str]:
        # Set scheduler
        self.set_scheduler(scheduler)
        
        # Clear any existing LoRA weights
        try:
            self.pipeline.unload_lora_weights()
        except Exception:
            pass  # No LoRAs loaded yet

        # Load LoRAs using native diffusers approach
        adapters = []
        weights = []
        
        for comp_name, cfg in composition.items():
            if isinstance(cfg, dict) and 'model_path' in cfg:
                weight = cfg.get("weight", 1.0)
                lora_path = self.download_lora(cfg['model_path'])
                
                if lora_path:
                    try:
                        # Use native diffusers LoRA loading
                        self.pipeline.load_lora_weights(
                            lora_path, 
                            adapter_name=comp_name
                        )
                        adapters.append(comp_name)
                        weights.append(weight)
                        logger.info(f"✅ Loaded LoRA {comp_name} with weight {weight}")
                    except Exception as e:
                        logger.error(f"❌ Failed to load LoRA {comp_name}: {e}")

        # Apply adapter composition if any were loaded
        if adapters:
            try:
                self.pipeline.set_adapters(adapters, adapter_weights=weights)
                logger.info(f"✅ Applied LoRA composition: {adapters} with weights {weights}")
            except Exception as e:
                logger.error(f"❌ Failed to set adapters: {e}")

        # Process ControlNet if specified
        control_image = None
        if control_type != ControlNetType.none and control_image_data:
            # Switch ControlNet if needed
            if control_type.value in self.controlnets:
                current_controlnet = self.controlnets[control_type.value]
                if self.pipeline.controlnet != current_controlnet:
                    logger.info(f"🔄 Switching to {control_type.value} ControlNet")
                    self.pipeline.controlnet = current_controlnet
                
                # Process control image
                control_image = self.process_control_image(control_image_data, control_type)
                if control_image:
                    logger.info(f"✅ Using {control_type.value} ControlNet with conditioning scale {controlnet_conditioning_scale}")
            else:
                logger.warning(f"⚠️ ControlNet {control_type.value} not available, using regular SDXL")

        # Set seed
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed else None

        # Generate images with or without ControlNet
        with torch.autocast("cuda" if self.device == "cuda" else "cpu"):
            if control_image is not None:
                # ControlNet generation
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=control_image,
                    num_inference_steps=steps,
                    guidance_scale=scale,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    width=width,
                    height=height,
                    num_images_per_prompt=num_images,
                    generator=generator
                )
            else:
                # Regular SDXL generation (ControlNet pipeline without control image)
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=scale,
                    width=width,
                    height=height,
                    num_images_per_prompt=num_images,
                    generator=generator
                )

        # Encode images to base64
        encoded = []
        for img in result.images:
            buf = BytesIO()
            img.save(buf, format="PNG")
            encoded.append(base64.b64encode(buf.getvalue()).decode())
        
        return encoded

@app.get("/ping")
async def ping():
    return {"status": "healthy"}

@app.post("/invocations")
async def invocations(req: GenerateRequest):
    try:
        if req.use_ai_prompts and req.composition:
            logger.info("Using Claude to generate prompt...")
            ai_prompts = composer.generate_intelligent_prompt(req.composition, req.character_description)
            prompt = req.prompt or ai_prompts.get("positive_prompt", "")
            negative = req.negative_prompt or ai_prompts.get("negative_prompt", "")
        else:
            prompt = req.prompt or "fantasy character, 8K, photorealistic"
            negative = req.negative_prompt or "blurry, deformed, bad anatomy"

        images = await composer.generate_images(
            composition=req.composition,
            prompt=prompt,
            negative_prompt=negative,
            steps=req.num_inference_steps,
            scale=req.guidance_scale,
            width=req.width,
            height=req.height,
            num_images=req.num_images,
            seed=req.seed,
            scheduler=req.scheduler,
            control_type=req.control_type,
            control_image_data=req.control_image,
            controlnet_conditioning_scale=req.controlnet_conditioning_scale
        )

        return JSONResponse(content={"images": images, "prompt": prompt, "composition_used": req.composition})

    except Exception as e:
        logger.exception("Inference error")
        return JSONResponse(status_code=500, content={"error": str(e)})

def get_default_model_bucket():
    try:
        sts = boto3.client("sts")
        acct = sts.get_caller_identity()["Account"]
        session = boto3.Session()
        region = session.region_name or "us-west-2"
        env = os.getenv("VIBEZ_ENV", "prod")
        suffix = "" if env == "prod" else f"-{env}"
        return f"vibez-model-registry{suffix}-{acct}-{region}"
    except:
        return "vibez-model-registry-796245059390-us-west-2"

def model_fn(model_dir=None):
    global composer
    composer = MultiLoRAComposer(os.getenv("MODEL_BUCKET", get_default_model_bucket()))
    return composer

# For local development, use: uvicorn inference:app --host 0.0.0.0 --port 8080
if __name__ == "__main__":
    import uvicorn
    model_fn()
    uvicorn.run(app, host="0.0.0.0", port=8080)
