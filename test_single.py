#!/usr/bin/env python3
"""Test single photography prompt with image saving"""

import json
import boto3
import time
import base64
from pathlib import Path

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')

def test_single_image():
    """Test a single photography prompt and save image"""
    
    payload = {
        "prompt": "Professional portrait of a young woman with long dark hair, soft natural window light, bokeh background, shot on Canon EOS R5 with RF 85mm f/1.2L lens at f/1.4, ISO 100, 1/200s shutter speed, warm color grading",
        "negative_prompt": "blurry, low quality, bad anatomy, ugly, distorted, oversaturated, underexposed",
        "num_images": 1,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "width": 1024,
        "height": 1024,
        "seed": 42
    }
    
    print("📸 Testing: Professional Portrait with Canon Settings")
    print(f"📝 Prompt: {payload['prompt']}")
    
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
            print(f"📐 Image size: {len(result['images'][0])} characters")
            
            # Create outputs directory
            outputs_dir = Path("outputs")
            outputs_dir.mkdir(exist_ok=True)
            
            # Save image
            img_data = base64.b64decode(result['images'][0])
            filename = f"portrait_canon_seed42.png"
            filepath = outputs_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            print(f"💾 Saved: {filepath}")
            print(f"📂 Check the {outputs_dir} directory for your image!")
            
        else:
            print("❌ No image generated")
            print(f"Response: {result}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_single_image()