#!/usr/bin/env python3
"""Single fashion test with perfect Franka settings"""

import json
import boto3
import time
import base64
from pathlib import Path

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')

def test_single_fashion():
    """Test single fashion image with perfect Franka settings from seed 4003"""
    
    payload = {
        "prompt": "Professional fashion photoshoot of caucasian franka person with long flowing hair, urban city street background, natural outdoor lighting, fashion modeling pose, contemporary street style, shot on Canon EOS R5 with 85mm lens, shallow depth of field, 100% photorealistic, commercial fashion photography, perfect caucasian skin tone, long beautiful hair",
        "negative_prompt": "blurry, low quality, bad anatomy, ugly, distorted, artificial, cartoon, CGI, plastic, unrealistic, bad hair, wig-like hair, asian, ethnic, dark skin, short hair, bob cut, pixie cut",
        "composition": {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",
                "weight": 1.3  # Maximum weight like successful seed 4003
            }
        },
        "num_images": 1,
        "num_inference_steps": 40,
        "guidance_scale": 6.5,
        "width": 1024,
        "height": 1024,
        "seed": 6001
    }
    
    print("📷 Testing Fashion Photoshoot with Perfect Franka Settings...")
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
            
            outputs_dir = Path("../_images/outputs")
            outputs_dir.mkdir(exist_ok=True)
            
            img_data = base64.b64decode(result['images'][0])
            filename = f"fashion_single_test_seed6001.png"
            filepath = outputs_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            print(f"💾 Saved: {filepath}")
            
        else:
            print("❌ No image generated")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_single_fashion()