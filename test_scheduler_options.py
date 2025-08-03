#!/usr/bin/env python3
"""Test different scheduler options including DPM++ 2M Karras"""

import json
import boto3
import time
import base64
from pathlib import Path

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')

def test_with_scheduler(prompt, scheduler_name, seed=42):
    """Test generation with specific scheduler"""
    
    payload = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, bad anatomy, ugly, distorted",
        "scheduler": scheduler_name,  # NEW PARAMETER!
        "num_images": 1,
        "num_inference_steps": 25,  # DPM++ 2M Karras needs fewer steps
        "guidance_scale": 7.0,
        "width": 1024,
        "height": 1024,
        "seed": seed
    }
    
    print(f"\n🔧 Testing with {scheduler_name} scheduler")
    print(f"📝 Prompt: {prompt}")
    
    try:
        start_time = time.time()
        response = runtime.invoke_endpoint(
            EndpointName='vibez-inference-endpoint',
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        result = json.loads(response['Body'].read())
        inference_time = time.time() - start_time
        
        if 'images' in result and result['images']:
            print(f"✅ Generated in {inference_time:.2f}s")
            
            outputs_dir = Path("../_images/outputs")
            outputs_dir.mkdir(exist_ok=True)
            
            img_data = base64.b64decode(result['images'][0])
            filename = f"{scheduler_name}_test_seed{seed}.png"
            filepath = outputs_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            print(f"💾 Saved: {filepath}")
            
        else:
            print("❌ No image generated")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def show_available_schedulers():
    """Show all available scheduler options"""
    
    print("🎛️  AVAILABLE SCHEDULERS:")
    print("=" * 60)
    print("- 'dpmpp_2m_karras' (default) - DPM++ 2M with Karras sigmas ⭐")
    print("- 'dpmpp_2m' - DPM++ 2M without Karras")
    print("- 'euler' - Euler discrete")
    print("- 'euler_a' - Euler ancestral (more random)")
    print("- 'ddim' - DDIM (deterministic)")
    print("- 'pndm' - PNDM (Pseudo Numerical)")
    print("- 'unipc' - UniPC multistep")
    print("=" * 60)

def main():
    """Test different schedulers"""
    
    show_available_schedulers()
    
    # Test prompt
    test_prompt = "Professional portrait of a woman, soft natural lighting, photorealistic, high quality"
    
    # Test with DPM++ 2M Karras (now the default!)
    test_with_scheduler(test_prompt, "dpmpp_2m_karras", seed=100)
    
    # Compare with Euler Ancestral
    test_with_scheduler(test_prompt, "euler_a", seed=100)
    
    # Example with ControlNet + LoRA + custom scheduler
    print("\n📋 EXAMPLE API PAYLOAD WITH SCHEDULER:")
    print("=" * 60)
    example = {
        "prompt": "Your prompt here",
        "scheduler": "dpmpp_2m_karras",  # ← NEW!
        "composition": {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",
                "weight": 1.0
            }
        },
        "control_image": "<base64_encoded_image>",
        "control_type": "openpose",
        "controlnet_conditioning_scale": 1.0,
        "num_images": 1,
        "num_inference_steps": 25,  # Can use fewer steps with DPM++ 2M Karras
        "guidance_scale": 7.0,
        "width": 1024,
        "height": 1024,
        "seed": 42
    }
    print(json.dumps(example, indent=2))

if __name__ == "__main__":
    main()