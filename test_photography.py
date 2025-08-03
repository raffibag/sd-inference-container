#!/usr/bin/env python3
"""Test photography-style prompts with SDXL"""

import json
import boto3
import time
import base64
import os
from pathlib import Path

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')

def test_prompt(prompt, test_name, seed=42):
    """Test a single photography prompt"""
    
    payload = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, bad anatomy, ugly, distorted, oversaturated, underexposed",
        "num_images": 1,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "width": 1024,
        "height": 1024,
        "seed": seed
    }
    
    print(f"\n📸 Testing: {test_name}")
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
            print(f"📐 Image size: {len(result['images'][0])} characters")
            
            # Save image to outputs directory
            outputs_dir = Path("outputs")
            outputs_dir.mkdir(exist_ok=True)
            
            for i, img_b64 in enumerate(result['images']):
                # Decode base64 image
                img_data = base64.b64decode(img_b64)
                
                # Create filename
                safe_name = test_name.lower().replace(" ", "_").replace("/", "_")
                filename = f"{safe_name}_seed{seed}_{i+1}.png"
                filepath = outputs_dir / filename
                
                # Save image
                with open(filepath, 'wb') as f:
                    f.write(img_data)
                
                print(f"💾 Saved: {filepath}")
        else:
            print("❌ No image generated")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def run_photography_tests():
    """Run a series of photography-style tests"""
    
    print("🎨 Running Photography Style Tests on GPU...")
    print("=" * 60)
    
    # Test 1: Portrait with specific camera settings
    test_prompt(
        "Professional portrait of a young woman with long dark hair, soft natural window light, "
        "bokeh background, shot on Canon EOS R5 with RF 85mm f/1.2L lens at f/1.4, "
        "ISO 100, 1/200s shutter speed, warm color grading",
        "Portrait with Canon Settings",
        seed=42
    )
    
    # Test 2: Landscape photography
    test_prompt(
        "Dramatic landscape photograph of mountains at golden hour, layered mountain ridges, "
        "atmospheric haze, shot on Fujifilm GFX 100S with GF 32-64mm lens at 45mm, "
        "f/8, ISO 100, circular polarizer filter, Velvia film simulation",
        "Landscape with Fujifilm Settings",
        seed=123
    )
    
    # Test 3: Street photography
    test_prompt(
        "Candid street photography in Tokyo at night, neon lights reflecting on wet pavement, "
        "silhouetted figure with umbrella, shot on Leica Q2 at 28mm, f/2.8, "
        "ISO 3200, 1/125s, high contrast black and white processing",
        "Street Photography B&W",
        seed=456
    )
    
    # Test 4: Product photography
    test_prompt(
        "Luxury watch product photography, Rolex Submariner on black velvet background, "
        "dramatic side lighting with subtle reflections, shot on Hasselblad H6D-100c "
        "with HC 120mm Macro lens, f/16, focus stacking, professional retouching",
        "Product Photography Macro",
        seed=789
    )
    
    # Test 5: Wildlife photography
    test_prompt(
        "Wildlife photograph of a majestic eagle in flight, wings spread wide, "
        "sharp detail on feathers, blurred forest background, shot on Nikon Z9 "
        "with 600mm f/4 lens, f/5.6, ISO 800, 1/2000s shutter speed, eye-tracking AF",
        "Wildlife Action Shot",
        seed=1011
    )
    
    print("\n" + "=" * 60)
    print("✅ Photography tests completed!")

if __name__ == "__main__":
    run_photography_tests()