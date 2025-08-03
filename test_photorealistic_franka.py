#!/usr/bin/env python3
"""Test photorealistic Franka with natural lighting, no style LoRA"""

import json
import boto3
import time
import base64
from pathlib import Path

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')

def test_photorealistic_franka(control_image_path, prompt, control_type, conditioning_scale, test_name, seed=42):
    """Test photorealistic Franka with just character LoRA"""
    
    control_image_b64 = image_to_base64(control_image_path)
    
    payload = {
        "prompt": prompt,
        "negative_prompt": "artistic, stylized, painting, illustration, cartoon, anime, CGI, 3D render, oversaturated, high contrast, dramatic lighting, artificial, plastic, unrealistic skin, makeup heavy, fashion editorial, avant-garde, abstract",
        "composition": {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",
                "weight": 0.9  # Moderate weight for natural look
            }
        },
        "control_image": control_image_b64,
        "control_type": control_type,
        "controlnet_conditioning_scale": conditioning_scale,
        "num_images": 1,
        "num_inference_steps": 35,
        "guidance_scale": 6.0,  # Moderate guidance for natural look
        "width": 1024,
        "height": 1024,
        "seed": seed
    }
    
    print(f"\n📸 Testing: {test_name}")
    print(f"🎛️  Control: {control_type} | Conditioning: {conditioning_scale}")
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
            safe_name = test_name.lower().replace(" ", "_").replace("/", "_")
            filename = f"photorealistic_{safe_name}_seed{seed}.png"
            filepath = outputs_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            print(f"💾 Saved: {filepath}")
            
        else:
            print("❌ No image generated")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def run_photorealistic_tests():
    """Test photorealistic Franka with natural lighting"""
    
    print("📷 Testing Photorealistic Franka with Natural Lighting...")
    print("=" * 80)
    
    control_dir = Path("../_images/control")
    
    # Test 1: Natural window light portrait
    test_photorealistic_franka(
        control_dir / "sin_versace_22.jpg",
        "Natural portrait photograph of franka person, soft window light, genuine smile, natural skin texture, realistic lighting, shot on Canon 5D Mark IV with 85mm lens, shallow depth of field, warm natural tones",
        "canny",
        1.2,  # Moderate control
        "Natural Window Light v22",
        seed=2001
    )
    
    # Test 2: Outdoor natural light
    test_photorealistic_franka(
        control_dir / "sin_versace_77.jpg", 
        "Outdoor portrait of franka person, golden hour natural lighting, soft shadows, realistic skin, candid expression, photographed with professional camera, natural color grading, photojournalistic style",
        "openpose",
        1.1,  # Light control for natural pose
        "Outdoor Golden Hour v77",
        seed=2002
    )
    
    # Test 3: Indoor soft lighting
    test_photorealistic_franka(
        control_dir / "sin_versace_22.jpg",
        "Indoor portrait of franka person, soft diffused lighting, natural expression, realistic skin texture, subtle shadows, shot with 50mm lens, natural color temperature, documentary photography style",
        "openpose", 
        1.0,  # Standard control
        "Indoor Soft Light v22",
        seed=2003
    )
    
    # Test 4: Professional headshot style
    test_photorealistic_franka(
        control_dir / "sin_versace_77.jpg",
        "Professional headshot of franka person, even lighting, confident expression, natural makeup, realistic skin, clean background, corporate photography, shot on medium format camera",
        "canny",
        1.3,  # Slightly stronger for clean composition
        "Professional Headshot v77", 
        seed=2004
    )
    
    # Test 5: Casual natural portrait
    test_photorealistic_franka(
        control_dir / "sin_versace_22.jpg",
        "Casual portrait of franka person, natural daylight, relaxed expression, minimal makeup, realistic lighting, lifestyle photography, authentic moment, natural skin tones",
        "openpose",
        0.9,  # Light control for natural feel
        "Casual Natural v22",
        seed=2005
    )
    
    print("\n" + "=" * 80) 
    print("✅ Photorealistic tests completed!")
    print("📷 All images should look natural and photorealistic without stylization")

if __name__ == "__main__":
    run_photorealistic_tests()