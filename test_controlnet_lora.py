#!/usr/bin/env python3
"""Test ControlNet with LoRA models using control images"""

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

def test_controlnet_lora(control_image_path, prompt, composition, control_type, test_name, seed=42):
    """Test ControlNet + LoRA with detailed photography settings"""
    
    # Convert control image to base64
    control_image_b64 = image_to_base64(control_image_path)
    
    payload = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, bad anatomy, ugly, distorted, oversaturated, underexposed, bad lighting, plastic skin, artificial, cartoon, deformed hands, extra fingers",
        "composition": composition,
        "control_image": control_image_b64,
        "control_type": control_type,
        "controlnet_conditioning_scale": 1.0,
        "num_images": 1,
        "num_inference_steps": 35,
        "guidance_scale": 7.5,
        "width": 1024,
        "height": 1024,
        "seed": seed
    }
    
    print(f"\n📸 Testing: {test_name}")
    print(f"🎛️  Control: {control_type} from {control_image_path.name}")
    print(f"📝 Prompt: {prompt}")
    print(f"🎭 Composition: {json.dumps(composition, indent=2)}")
    
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
            
            # Save to custom outputs directory
            outputs_dir = Path("../_images/outputs")
            outputs_dir.mkdir(exist_ok=True)
            
            # Save image
            img_data = base64.b64decode(result['images'][0])
            safe_name = test_name.lower().replace(" ", "_").replace("/", "_")
            filename = f"controlnet_{safe_name}_seed{seed}.png"
            filepath = outputs_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            print(f"💾 Saved: {filepath}")
            
        else:
            print("❌ No image generated")
            print(f"Response: {result}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def run_controlnet_lora_tests():
    """Run ControlNet + LoRA tests with the provided control images"""
    
    print("🎨 Running ControlNet + LoRA Tests with Control Images...")
    print("=" * 80)
    
    control_dir = Path("../_images/control")
    
    # Test 1: sin_versace_22.jpg with Franka + Sinisha (Canny)
    test_controlnet_lora(
        control_dir / "sin_versace_22.jpg",
        "Professional fashion portrait of franka person in elegant black dress, dramatic studio lighting, "
        "high fashion editorial style, shot on Hasselblad X2D 100C with XCD 80mm f/1.9 lens, "
        "aperture f/2.8, ISO 100, 1/125s, professional makeup, confident pose, "
        "fashion magazine quality, contemporary styling",
        {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",
                "weight": 1.2
            },
            "style": {
                "model_path": "auto-2025-01-31-17-52-39",
                "weight": 0.9
            }
        },
        "canny",
        "Franka Sinisha Fashion Canny v22",
        seed=600
    )
    
    # Test 2: sin_versace_77.jpg with Franka + Sinisha (Canny)
    test_controlnet_lora(
        control_dir / "sin_versace_77.jpg",
        "Editorial portrait of franka person, avant-garde fashion styling, dramatic chiaroscuro lighting, "
        "shot on Canon EOS R5 with RF 85mm f/1.2L lens, aperture f/1.8, ISO 200, 1/160s, "
        "professional retouching, high contrast black and white conversion, "
        "contemporary fashion aesthetic, artistic composition",
        {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",
                "weight": 1.1
            },
            "style": {
                "model_path": "auto-2025-01-31-17-52-39",
                "weight": 0.8
            }
        },
        "canny",
        "Franka Sinisha Editorial Canny v77",
        seed=700
    )
    
    # Test 3: sin_versace_22.jpg with OpenPose (different approach)
    test_controlnet_lora(
        control_dir / "sin_versace_22.jpg",
        "Luxury portrait of franka person in sophisticated black attire, soft directional lighting, "
        "shot on Sony A7R V with FE 50mm f/1.2 GM lens, aperture f/2, ISO 100, 1/200s, "
        "medium format look, precise focus on eyes, elegant posing, "
        "high-end fashion photography, commercial beauty standards",
        {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",
                "weight": 1.0
            },
            "style": {
                "model_path": "auto-2025-01-31-17-52-39",
                "weight": 0.7
            }
        },
        "openpose",
        "Franka Sinisha Luxury OpenPose v22",
        seed=800
    )
    
    # Test 4: sin_versace_77.jpg with OpenPose
    test_controlnet_lora(
        control_dir / "sin_versace_77.jpg",
        "High fashion beauty shot of franka person, professional studio setup with key light and fill, "
        "shot on Fujifilm GFX 100S with GF 110mm f/2 lens, aperture f/2.8, ISO 64, 1/125s, "
        "tethered shooting workflow, perfect skin texture, editorial makeup, "
        "contemporary beauty photography, magazine cover quality",
        {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",
                "weight": 1.15
            },
            "style": {
                "model_path": "auto-2025-01-31-17-52-39",
                "weight": 0.85
            }
        },
        "openpose",
        "Franka Sinisha Beauty OpenPose v77",
        seed=900
    )
    
    print("\n" + "=" * 80)
    print("✅ ControlNet + LoRA tests completed!")
    print("📂 Check ../_images/outputs/ for all generated images")

if __name__ == "__main__":
    run_controlnet_lora_tests()