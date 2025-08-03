#!/usr/bin/env python3
"""Test SDXL with LoRA models using detailed photography prompts"""

import json
import boto3
import time
import base64
from pathlib import Path

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')

def test_lora_prompt(prompt, composition, test_name, seed=42):
    """Test a LoRA prompt with detailed photography settings"""
    
    payload = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, bad anatomy, ugly, distorted, oversaturated, underexposed, bad lighting, plastic skin, artificial, cartoon",
        "composition": composition,
        "num_images": 1,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "width": 1024,
        "height": 1024,
        "seed": seed
    }
    
    print(f"\n📸 Testing: {test_name}")
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
            
            # Create custom outputs directory
            outputs_dir = Path("/Users/rafbagdasarian/newOS/sd-inference-container/lora_test_outputs")
            outputs_dir.mkdir(exist_ok=True)
            
            # Save image
            img_data = base64.b64decode(result['images'][0])
            safe_name = test_name.lower().replace(" ", "_").replace("/", "_")
            filename = f"{safe_name}_seed{seed}.png"
            filepath = outputs_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            print(f"💾 Saved: {filepath}")
            
        else:
            print("❌ No image generated")
            print(f"Response: {result}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def run_lora_photography_tests():
    """Run detailed photography tests with LoRA models"""
    
    print("🎨 Running LoRA Photography Tests with Detailed Camera Settings...")
    print("=" * 80)
    
    # Test 1: Franka Portrait - Studio lighting with 50mm
    test_lora_prompt(
        "Professional studio portrait of franka person, soft diffuse lighting setup with key light at 45 degrees, "
        "subtle rim lighting, seamless white backdrop, shot on Canon EOS R6 Mark II with RF 50mm f/1.2L lens, "
        "aperture f/2.8 for optimal sharpness, ISO 200, 1/160s shutter speed, color temperature 5600K, "
        "professional makeup, natural expression, eyes sharp and well-lit",
        {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",
                "weight": 1.0
            }
        },
        "Franka Studio Portrait 50mm",
        seed=100
    )
    
    # Test 2: Franka with Sinisha Style - 25mm wide angle environmental
    test_lora_prompt(
        "Environmental portrait of franka person in modern architectural space, wide angle perspective, "
        "natural window light mixed with architectural lighting, shot on Sony A7R V with FE 24-70mm f/2.8 GM II at 25mm, "
        "aperture f/4 for environmental context, ISO 400, 1/125s, focus on subject with sharp background detail, "
        "contemporary styling, confident pose, geometric composition",
        {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",
                "weight": 0.9
            },
            "style": {
                "model_path": "auto-2025-01-31-17-52-39",
                "weight": 0.8
            }
        },
        "Franka + Sinisha 25mm Environmental",
        seed=200
    )
    
    # Test 3: Franka Beauty Shot - 85mm with dramatic lighting
    test_lora_prompt(
        "High-end beauty portrait of franka person, dramatic Rembrandt lighting with single large softbox, "
        "subtle fill light, hair light for separation, shot on Fujifilm GFX 100S with GF 110mm f/2 lens, "
        "aperture f/2.8, ISO 100, 1/200s, tethered shooting, perfect skin texture, glossy lips, "
        "professional retouching look, commercial beauty standards",
        {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",
                "weight": 1.1
            }
        },
        "Franka Beauty 85mm Dramatic",
        seed=300
    )
    
    # Test 4: Franka Outdoor - Golden hour with 35mm
    test_lora_prompt(
        "Outdoor lifestyle portrait of franka person during golden hour, natural backlighting creating rim light, "
        "reflector for gentle fill light on face, shot on Nikon Z8 with NIKKOR Z 35mm f/1.8 S lens, "
        "aperture f/2.8, ISO 250, 1/250s, auto white balance with slight warm bias, "
        "natural wind-blown hair, candid expression, urban background with bokeh",
        {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",
                "weight": 0.95
            }
        },
        "Franka Outdoor Golden Hour 35mm",
        seed=400
    )
    
    # Test 5: Pure Sinisha Style - Fashion editorial
    test_lora_prompt(
        "High fashion editorial portrait, avant-garde lighting setup with colored gels, "
        "dramatic shadows and highlights, shot on Hasselblad X2D 100C with XCD 80mm f/1.9 lens, "
        "aperture f/2.8, ISO 64, 1/125s, medium format depth and detail, "
        "editorial makeup and styling, contemporary fashion aesthetic, artistic composition",
        {
            "style": {
                "model_path": "auto-2025-01-31-17-52-39",
                "weight": 1.0
            }
        },
        "Pure Sinisha Fashion Editorial",
        seed=500
    )
    
    print("\n" + "=" * 80)
    print("✅ LoRA Photography tests completed!")
    print("📂 Check the lora_test_outputs/ directory for all generated images")

if __name__ == "__main__":
    run_lora_photography_tests()