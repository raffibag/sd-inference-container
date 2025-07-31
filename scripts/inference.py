#!/usr/bin/env python
"""
Multi-LoRA Composition Inference for Character Generation
Supports dynamic loading and blending of multiple LoRA models
"""

import json
import os
import boto3
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
import logging
from typing import Dict, List, Optional
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class MultiLoRAComposer:
    def __init__(self, model_bucket: str, base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        self.model_bucket = model_bucket
        self.base_model = base_model
        self.s3_client = boto3.client('s3')
        self.bedrock_client = boto3.client('bedrock-runtime')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lora_cache = {}
        
        # Load SDXL base pipeline
        from diffusers import StableDiffusionXLPipeline
        logger.info(f"Loading SDXL base model: {base_model}")
        self.base_pipeline = StableDiffusionXLPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if self.device == "cuda" else None
        )
        self.base_pipeline = self.base_pipeline.to(self.device)
        self.base_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.base_pipeline.scheduler.config
        )
        
        # Enable memory optimizations
        if self.device == "cuda":
            self.base_pipeline.enable_xformers_memory_efficient_attention()
            self.base_pipeline.enable_model_cpu_offload()
    
    def load_model_registry(self) -> Dict:
        """Load the model registry from S3"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.model_bucket,
                Key="registry/models.json"
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            logger.warning(f"Could not load model registry: {e}")
            return {}
    
    def download_lora_model(self, model_path: str) -> str:
        """Download LoRA model from S3 if not cached"""
        local_path = f"/tmp/lora_models/{model_path}"
        
        if model_path in self.lora_cache:
            return self.lora_cache[model_path]
        
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        try:
            # Download LoRA weights
            self.s3_client.download_file(
                self.model_bucket,
                f"models/{model_path}/pytorch_lora_weights.safetensors",
                f"{local_path}/pytorch_lora_weights.safetensors"
            )
            
            # Download adapter config
            self.s3_client.download_file(
                self.model_bucket,
                f"models/{model_path}/adapter_config.json",
                f"{local_path}/adapter_config.json"
            )
            
            self.lora_cache[model_path] = local_path
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download LoRA model {model_path}: {e}")
            return None
    
    def generate_intelligent_prompt(self, composition: Dict, character_description: str = "") -> Dict[str, str]:
        """Use Claude to generate optimized prompts based on composition"""
        try:
            # Build context about the composition
            context_parts = []
            trigger_words = []
            
            model_registry = self.load_model_registry()
            
            for component_type, config in composition.items():
                if isinstance(config, dict) and 'model_path' in config:
                    model_path = config['model_path']
                    weight = config.get('weight', 1.0)
                    
                    # Find model info in registry
                    for project, project_data in model_registry.get('projects', {}).items():
                        for model_name, model_info in project_data.get('models', {}).items():
                            if model_info.get('path') == model_path:
                                trigger_word = model_info.get('trigger_word', '')
                                if trigger_word:
                                    trigger_words.append(trigger_word)
                                
                                context_parts.append(f"{component_type}: {trigger_word} (weight: {weight})")
                                
                                metadata = model_info.get('metadata', {})
                                if metadata:
                                    context_parts.append(f"  - Style: {metadata.get('style', 'N/A')}")
                                    if 'dominant_colors' in metadata:
                                        context_parts.append(f"  - Colors: {', '.join(metadata['dominant_colors'])}")
            
            # Create Claude prompt
            claude_prompt = f"""You are an expert Stable Diffusion XL prompt engineer. Create optimized prompts for generating high-quality fantasy characters.

Character Description: {character_description or 'Epic fantasy character'}

Composition Details:
{chr(10).join(context_parts)}

Required trigger words to include: {', '.join(trigger_words)}

Please generate:
1. A detailed positive prompt (focusing on visual quality, composition, lighting, and artistic style)
2. A comprehensive negative prompt (to avoid common issues like blurriness, deformities, etc.)

Guidelines:
- Include all trigger words naturally
- Focus on photorealistic quality for SDXL
- Add professional photography terms
- Include lighting and composition details
- Emphasize high resolution and detail
- Make it fantasy-appropriate but realistic

Return in this exact JSON format:
{{"positive_prompt": "...", "negative_prompt": "..."}}"""

            # Call Claude via Bedrock
            response = self.bedrock_client.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": claude_prompt
                        }
                    ]
                })
            )
            
            response_body = json.loads(response['body'].read())
            claude_text = response_body['content'][0]['text']
            
            # Extract JSON from Claude response
            import re
            json_match = re.search(r'\{.*\}', claude_text, re.DOTALL)
            if json_match:
                prompt_data = json.loads(json_match.group())
                return {
                    "positive_prompt": prompt_data.get("positive_prompt", ""),
                    "negative_prompt": prompt_data.get("negative_prompt", "")
                }
            
        except Exception as e:
            logger.error(f"Claude prompt generation failed: {e}")
        
        # Fallback to basic prompt generation
        basic_prompt = f"high quality, detailed, {', '.join(trigger_words)}, professional photography, 8K resolution"
        basic_negative = "blurry, low quality, deformed, ugly, bad anatomy, missing limbs, extra limbs, watermark"
        
        return {
            "positive_prompt": basic_prompt,
            "negative_prompt": basic_negative
        }

    def compose_character(self, composition_config: Dict) -> StableDiffusionXLPipeline:
        """Create a composed pipeline with multiple LoRAs"""
        from diffusers import StableDiffusionXLPipeline
        
        # Start with base SDXL pipeline copy
        pipeline = StableDiffusionXLPipeline(
            text_encoder=self.base_pipeline.text_encoder,
            text_encoder_2=self.base_pipeline.text_encoder_2,
            vae=self.base_pipeline.vae,
            unet=self.base_pipeline.unet,
            tokenizer=self.base_pipeline.tokenizer,
            tokenizer_2=self.base_pipeline.tokenizer_2,
            scheduler=self.base_pipeline.scheduler
        )
        
        # Apply LoRAs with weights
        for component, config in composition_config.items():
            if isinstance(config, dict) and 'model_path' in config:
                model_path = config['model_path']
                weight = config.get('weight', 1.0)
                
                logger.info(f"Loading LoRA: {model_path} with weight {weight}")
                
                local_path = self.download_lora_model(model_path)
                if local_path:
                    try:
                        # Load LoRA into UNet
                        pipeline.unet = PeftModel.from_pretrained(
                            pipeline.unet,
                            local_path,
                            adapter_name=component
                        )
                        
                        # Set adapter weight
                        pipeline.unet.set_adapter(component)
                        pipeline.unet.set_adapter_weights(component, weight)
                        
                    except Exception as e:
                        logger.error(f"Failed to load LoRA {model_path}: {e}")
        
        return pipeline
    
    def generate_character(
        self,
        composition: Dict,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        num_images: int = 1,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """Generate character images with composed LoRAs"""
        
        # Create composed pipeline
        pipeline = self.compose_character(composition)
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate images
        with torch.autocast("cuda" if self.device == "cuda" else "cpu"):
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                num_images_per_prompt=num_images
            )
        
        return result.images

# Global composer instance
composer = None

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "healthy"})

@app.route('/invocations', methods=['POST'])
def invocations():
    try:
        data = request.get_json()
        
        # Parse request
        composition = data.get('composition', {})
        character_description = data.get('character_description', '')
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative_prompt', '')
        use_ai_prompts = data.get('use_ai_prompts', True)
        num_inference_steps = data.get('num_inference_steps', 30)  # Higher for SDXL
        guidance_scale = data.get('guidance_scale', 7.5)
        width = data.get('width', 1024)  # SDXL native resolution
        height = data.get('height', 1024)
        num_images = data.get('num_images', 1)
        seed = data.get('seed')
        
        # Generate AI-optimized prompts if requested
        if use_ai_prompts and composition:
            logger.info("Generating AI-optimized prompts...")
            ai_prompts = composer.generate_intelligent_prompt(composition, character_description)
            
            # Use AI prompts if user didn't provide custom ones
            if not prompt:
                prompt = ai_prompts['positive_prompt']
            if not negative_prompt:
                negative_prompt = ai_prompts['negative_prompt']
                
            logger.info(f"Generated prompt: {prompt[:100]}...")
        
        # Generate images
        images = composer.generate_character(
            composition=composition,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            num_images=num_images,
            seed=seed
        )
        
        # Convert images to base64
        image_data = []
        for img in images:
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            image_data.append(img_str)
        
        return jsonify({
            "images": image_data,
            "composition_used": composition
        })
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return jsonify({"error": str(e)}), 500

def get_default_model_bucket():
    """Generate dynamic bucket name based on AWS account and region"""
    try:
        # Get AWS account ID and region from boto3
        sts_client = boto3.client('sts')
        account_id = sts_client.get_caller_identity()['Account']
        
        session = boto3.Session()
        region = session.region_name or 'us-west-2'
        
        # Get environment (defaults to prod)
        env = os.environ.get('VIBEZ_ENV', 'prod')
        env_suffix = '' if env == 'prod' else f'-{env}'
        
        return f'vibez-model-registry{env_suffix}-{account_id}-{region}'
    except Exception as e:
        # Fallback to current bucket if AWS calls fail
        return 'vibez-model-registry-796245059390-us-west-2'

def model_fn(model_dir):
    """Load model for SageMaker"""
    global composer
    model_bucket = os.environ.get('MODEL_BUCKET', get_default_model_bucket())
    composer = MultiLoRAComposer(model_bucket)
    return composer

if __name__ == '__main__':
    # Initialize for local testing
    model_bucket = os.environ.get('MODEL_BUCKET', get_default_model_bucket())
    composer = MultiLoRAComposer(model_bucket)
    
    # Start Flask app
    app.run(host='0.0.0.0', port=8080)