#!/usr/bin/env python3
"""Custom prompt testing with ControlNet - shows exact command usage"""

import json
import boto3
import time
import base64
from pathlib import Path
import sys

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')

def test_custom_controlnet(control_image_path, prompt, control_type="openpose", use_lora=True):
    """Test custom prompt with ControlNet"""
    
    # Log the command used to run this script
    print("🖥️  COMMAND TO RUN THIS SCRIPT:")
    print("=" * 80)
    print(f"AWS_PROFILE=raffibag python3 {sys.argv[0]}")
    print("=" * 80)
    print()
    
    # Convert control image to base64
    control_image_b64 = image_to_base64(control_image_path)
    
    # Build payload
    payload = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, bad anatomy, ugly, distorted, oversaturated, underexposed",
        "control_image": control_image_b64,
        "control_type": control_type,
        "controlnet_conditioning_scale": 1.0,
        "num_images": 1,
        "num_inference_steps": 35,
        "guidance_scale": 7.0,
        "width": 1024,
        "height": 1024,
        "seed": 42
    }
    
    # Add LoRA if requested
    if use_lora:
        payload["composition"] = {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",  # Franka
                "weight": 1.0
            }
        }
    
    print(f"📸 Testing Custom Prompt with ControlNet")
    print(f"🎛️  Control: {control_type} from {control_image_path.name}")
    print(f"📝 Prompt: {prompt}")
    print(f"🎭 Using LoRA: {use_lora}")
    print()
    
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
            timestamp = int(time.time())
            filename = f"custom_prompt_{timestamp}.png"
            filepath = outputs_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            print(f"💾 Saved: {filepath}")
            
        else:
            print("❌ No image generated")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure the endpoint is deployed!")

def main():
    """Main function - EDIT YOUR PROMPT HERE"""
    
    # EDIT THESE VARIABLES FOR YOUR TEST
    # ===================================
    
    # Your custom prompt
    MY_PROMPT = "Professional portrait of a woman in elegant dress, studio lighting"
    
    # Control image to use (from ../_images/control/)
    CONTROL_IMAGE = "sin_versace_22.jpg"
    
    # Control type: "canny", "openpose", or "depth"
    CONTROL_TYPE = "openpose"
    
    # Use Franka LoRA? (True/False)
    USE_LORA = True
    
    # ===================================
    
    control_path = Path("../_images/control") / CONTROL_IMAGE
    
    if not control_path.exists():
        print(f"❌ Control image not found: {control_path}")
        print(f"Available control images:")
        for img in Path("../_images/control").glob("*.jpg"):
            print(f"  - {img.name}")
        return
    
    test_custom_controlnet(control_path, MY_PROMPT, CONTROL_TYPE, USE_LORA)

if __name__ == "__main__":
    main()