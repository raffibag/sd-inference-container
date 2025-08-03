#!/usr/bin/env python3
"""Simple test for SDXL inference endpoint"""

import json
import boto3

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')

def test_inference():
    """Test basic SDXL inference without LoRA"""
    
    # Simple test payload - just SDXL, no LoRA, no ControlNet
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
    
    print("🚀 Testing SDXL inference endpoint (no LoRA, no ControlNet)...")
    print(f"📝 Prompt: {payload['prompt']}")
    print(f"🎛️  Steps: {payload['num_inference_steps']}, Guidance: {payload['guidance_scale']}")
    
    try:
        # Invoke endpoint
        print("\n⏳ Invoking endpoint...")
        response = runtime.invoke_endpoint(
            EndpointName='vibez-inference-endpoint',
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Parse response
        result = json.loads(response['Body'].read())
        
        if 'images' in result and result['images']:
            print(f"\n✅ SUCCESS! Generated {len(result['images'])} image(s)")
            print(f"📐 Image data length: {len(result['images'][0])} characters")
            
            # Print response details
            if 'parameters' in result:
                print(f"\n📊 Parameters used:")
                for k, v in result['parameters'].items():
                    print(f"   - {k}: {v}")
                    
        else:
            print("\n❌ No images in response")
            print(f"Response: {json.dumps(result, indent=2)}")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Response body: {e.response}")

if __name__ == "__main__":
    test_inference()