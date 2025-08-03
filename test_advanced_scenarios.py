#!/usr/bin/env python3
"""Test face swap, wide shots, and fantasy creatures with Franka LoRA"""

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

def test_scenario(control_image_path, prompt, composition, control_type, conditioning_scale, test_name, seed=42):
    """Test various advanced scenarios"""
    
    # Only use control image if provided
    payload = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, bad anatomy, ugly, distorted, artificial, cartoon, CGI, plastic, unrealistic",
        "num_images": 1,
        "num_inference_steps": 35,
        "guidance_scale": 6.0,
        "width": 1024,
        "height": 1024,
        "seed": seed
    }
    
    # Add composition if provided
    if composition:
        payload["composition"] = composition
    
    # Add control image if provided
    if control_image_path:
        control_image_b64 = image_to_base64(control_image_path)
        payload["control_image"] = control_image_b64
        payload["control_type"] = control_type
        payload["controlnet_conditioning_scale"] = conditioning_scale
    
    print(f"\n📸 Testing: {test_name}")
    if control_image_path:
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
            filename = f"advanced_{safe_name}_seed{seed}.png"
            filepath = outputs_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            print(f"💾 Saved: {filepath}")
            
        else:
            print("❌ No image generated")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def run_advanced_tests():
    """Test face swap, wide shots, and fantasy scenarios"""
    
    print("🎭 Testing Advanced Scenarios...")
    print("=" * 80)
    
    control_dir = Path("../_images/control")
    franka_composition = {
        "character": {
            "model_path": "auto-2025-01-31-18-05-24",
            "weight": 0.9
        }
    }
    
    # 1. Face Swap - Using successful OpenPose settings
    test_scenario(
        control_dir / "sin_versace_22.jpg",
        "Natural portrait photograph of franka person, soft natural lighting, realistic skin texture, documentary photography style, candid expression, lifestyle photography, natural makeup",
        franka_composition,
        "openpose",
        0.9,  # Light control like successful seed 2005
        "Face Swap Natural v22",
        seed=3001
    )
    
    test_scenario(
        control_dir / "sin_versace_77.jpg", 
        "Indoor portrait of franka person, soft diffused lighting, natural expression, realistic skin texture, documentary photography style, minimal makeup, lifestyle photography",
        franka_composition,
        "openpose",
        1.0,  # Like successful seed 2003
        "Face Swap Natural v77",
        seed=3002
    )
    
    # 2. Wide Shot Downtown LA
    test_scenario(
        None,  # No control image
        "Wide shot photograph of franka person walking down busy street in downtown Los Angeles, urban landscape, city skyline background, natural daylight, street photography, candid moment, realistic lighting, shot with 24mm lens, environmental portrait",
        franka_composition,
        None, None,
        "Wide Shot Downtown LA Walking",
        seed=3003
    )
    
    test_scenario(
        None,  # No control image  
        "Wide environmental portrait of franka person in downtown Los Angeles financial district, modern skyscrapers background, golden hour lighting, urban lifestyle photography, shot with 35mm lens, full body composition, natural posing",
        franka_composition,
        None, None,
        "Wide Shot LA Financial District",
        seed=3004
    )
    
    # 3. Fantasy Creature - Manta Ray with Franka features
    test_scenario(
        None,  # No control image
        "Underwater photograph of a majestic manta ray swimming upwards in clear blue ocean water, the manta ray has delicate white bat wings instead of fins, small baby bat head with subtle facial features resembling franka person, ethereal underwater lighting, rays of sunlight filtering through water, marine biology photography, National Geographic style",
        {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24", 
                "weight": 0.6  # Lower weight for fantasy blend
            }
        },
        None, None,
        "Manta Ray Bat Franka Hybrid",
        seed=3005
    )
    
    test_scenario(
        None,  # No control image
        "Underwater scene of graceful manta ray with white bat wings swimming towards surface, creature has gentle facial features inspired by franka person, clear turquoise water, coral reef below, volumetric underwater lighting, marine life photography, magical realism",
        {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",
                "weight": 0.5  # Even lower for more fantasy
            }
        },
        None, None,
        "Manta Ray Fantasy Creature",
        seed=3006
    )
    
    print("\n" + "=" * 80)
    print("✅ Advanced scenario tests completed!")
    print("🎭 Check results: face swaps, LA wide shots, and fantasy creatures")

if __name__ == "__main__":
    run_advanced_tests()