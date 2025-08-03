#!/usr/bin/env python3
"""Test stronger ControlNet control with adjusted parameters"""

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

def test_strong_controlnet(control_image_path, prompt, composition, control_type, conditioning_scale, guidance_scale, test_name, seed=42):
    """Test ControlNet with stronger conditioning"""
    
    control_image_b64 = image_to_base64(control_image_path)
    
    payload = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, bad anatomy, ugly, distorted, oversaturated, underexposed, deformed hands, extra fingers, wrong proportions",
        "composition": composition,
        "control_image": control_image_b64,
        "control_type": control_type,
        "controlnet_conditioning_scale": conditioning_scale,
        "num_images": 1,
        "num_inference_steps": 40,  # More steps for better quality
        "guidance_scale": guidance_scale,
        "width": 1024,
        "height": 1024,
        "seed": seed
    }
    
    print(f"\n📸 Testing: {test_name}")
    print(f"🎛️  Control: {control_type} | Conditioning: {conditioning_scale} | Guidance: {guidance_scale}")
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
            filename = f"strong_{safe_name}_seed{seed}.png"
            filepath = outputs_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            print(f"💾 Saved: {filepath}")
            
        else:
            print("❌ No image generated")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def run_stronger_controlnet_tests():
    """Test different ControlNet strength configurations"""
    
    print("🎯 Testing Stronger ControlNet Control...")
    print("=" * 80)
    
    control_dir = Path("../_images/control")
    
    # Reduced LoRA weights to not fight with ControlNet
    base_composition = {
        "character": {
            "model_path": "auto-2025-01-31-18-05-24",
            "weight": 0.8  # Reduced from 1.1-1.2
        },
        "style": {
            "model_path": "auto-2025-01-31-17-52-39",
            "weight": 0.6  # Reduced from 0.8-0.9
        }
    }
    
    # Simpler prompt to let ControlNet dominate
    simple_prompt = "Professional portrait of franka person, elegant styling, studio lighting"
    
    # Test 1: Very Strong Canny Control
    test_strong_controlnet(
        control_dir / "sin_versace_22.jpg",
        simple_prompt,
        base_composition,
        "canny",
        1.8,  # conditioning_scale - Very strong
        5.0,  # guidance_scale - Lower guidance
        "Very Strong Canny v22",
        seed=1001
    )
    
    # Test 2: Very Strong OpenPose Control  
    test_strong_controlnet(
        control_dir / "sin_versace_77.jpg",
        simple_prompt,
        base_composition,
        "openpose",
        1.8,  # conditioning_scale - Very strong
        5.0,  # guidance_scale - Lower guidance
        "Very Strong OpenPose v77",
        seed=1002
    )
    
    # Test 3: Depth Control (often better for pose)
    test_strong_controlnet(
        control_dir / "sin_versace_22.jpg",
        simple_prompt,
        base_composition,
        "depth",
        1.5,  # conditioning_scale
        5.5,  # guidance_scale
        "Depth Control v22",
        seed=1003
    )
    
    # Test 4: Extreme ControlNet dominance
    test_strong_controlnet(
        control_dir / "sin_versace_77.jpg",
        "franka person portrait",  # Minimal prompt
        {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",
                "weight": 0.7  # Even lower
            }
        },
        "canny",
        2.0,  # conditioning_scale - Maximum strength
        4.0,  # guidance_scale - Very low guidance
        "Extreme Canny Control v77",
        seed=1004
    )
    
    # Test 5: Try different control image with depth
    test_strong_controlnet(
        control_dir / "sin_versace_77.jpg",
        simple_prompt,
        base_composition,
        "depth", 
        1.6,  # conditioning_scale
        5.0,  # guidance_scale
        "Strong Depth v77",
        seed=1005
    )
    
    print("\n" + "=" * 80)
    print("✅ Strong ControlNet tests completed!")
    print("📊 Compare results to see which settings preserve pose/composition best")

if __name__ == "__main__":
    run_stronger_controlnet_tests()