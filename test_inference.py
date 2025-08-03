#!/usr/bin/env python3
"""Test inference endpoint with GPU detection check"""

import json
import boto3
import base64
from io import BytesIO
from PIL import Image

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')

def test_inference():
    """Test basic inference and check logs for GPU detection"""
    
    # Simple test payload
    payload = {
        "prompt": "A professional portrait of a young woman, soft natural lighting, bokeh background, shot on Canon 5D Mark IV, 85mm f/1.4 lens",
        "negative_prompt": "blurry, low quality, bad anatomy, ugly, distorted",
        "num_images": 1,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "width": 1024,
        "height": 1024,
        "seed": 42
    }
    
    print("🚀 Testing inference endpoint...")
    print(f"📝 Prompt: {payload['prompt']}")
    
    try:
        # Invoke endpoint
        response = runtime.invoke_endpoint(
            EndpointName='sd-controlnet-lora-endpoint',
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Parse response
        result = json.loads(response['Body'].read())
        
        if 'images' in result and result['images']:
            print(f"✅ Successfully generated {len(result['images'])} image(s)")
            
            # Save first image
            img_data = base64.b64decode(result['images'][0])
            img = Image.open(BytesIO(img_data))
            img.save('test_output.png')
            print("💾 Saved test image to test_output.png")
            
            # Print response details
            print("\n📊 Response details:")
            print(f"- Composition: {result.get('composition', {})}")
            print(f"- Parameters: {result.get('parameters', {})}")
            
        else:
            print("❌ No images in response")
            print(f"Response: {result}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        
    # Check CloudWatch logs for GPU detection
    print("\n🔍 Check CloudWatch logs for GPU detection:")
    print("AWS_PROFILE=raffibag aws logs tail /aws/sagemaker/Endpoints/sd-controlnet-lora-endpoint --follow")

if __name__ == "__main__":
    test_inference()