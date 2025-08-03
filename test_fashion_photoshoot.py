#!/usr/bin/env python3
"""Fashion photoshoot with perfect Franka likeness and cute sea creature"""

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

def test_fashion_scenario(control_image_path, prompt, composition, control_type, conditioning_scale, test_name, seed=42):
    """Test fashion photoshoot scenarios with perfect Franka settings"""
    
    payload = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, bad anatomy, ugly, distorted, artificial, cartoon, CGI, plastic, unrealistic, bad hair, wig-like hair, asian, ethnic, dark skin, short hair, bob cut, pixie cut",
        "num_images": 1,
        "num_inference_steps": 40,  # High quality
        "guidance_scale": 6.5,
        "width": 1024,
        "height": 1024,
        "seed": seed
    }
    
    # Add composition
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
            filename = f"fashion_{safe_name}_seed{seed}.png"
            filepath = outputs_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            print(f"💾 Saved: {filepath}")
            
        else:
            print("❌ No image generated")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def run_fashion_photoshoot_tests():
    """Test fashion photoshoot and cute sea creature"""
    
    print("📷 Testing Fashion Photoshoot with Perfect Franka Settings...")
    print("=" * 80)
    
    control_dir = Path("../_images/control")
    
    # Perfect Franka composition based on successful seed 4003
    perfect_franka_composition = {
        "character": {
            "model_path": "auto-2025-01-31-18-05-24",
            "weight": 1.3  # Maximum weight like successful seed 4003
        }
    }
    
    # 1. Urban Fashion - Controlled pose
    test_fashion_scenario(
        control_dir / "sin_versace_22.jpg",
        "Professional fashion photoshoot of caucasian franka person with long flowing hair, urban city street background, natural outdoor lighting, fashion modeling pose, contemporary street style, shot on Canon EOS R5 with 85mm lens, shallow depth of field, 100% photorealistic, commercial fashion photography, perfect caucasian skin tone, long beautiful hair",
        perfect_franka_composition,
        "openpose",
        0.8,  # Low control like successful seed 4003
        "Urban Fashion Controlled v22",
        seed=5001
    )
    
    # 2. Urban Fashion - Different pose
    test_fashion_scenario(
        control_dir / "sin_versace_77.jpg",
        "High-end fashion photoshoot of caucasian franka person with long hair, downtown city environment, golden hour natural lighting, professional modeling, elegant fashion pose, shot with medium format camera, fashion magazine quality, 100% realistic photography, caucasian features, long flowing hair style",
        perfect_franka_composition,
        "openpose", 
        0.8,  # Low control
        "Urban Fashion Controlled v77",
        seed=5002
    )
    
    # 3. City Fashion - No control (natural composition)
    test_fashion_scenario(
        None,  # No control image
        "Professional fashion photoshoot of caucasian franka person with long beautiful hair, modern city skyline background, natural outdoor lighting, confident modeling pose, contemporary fashion, shot on Hasselblad X2D with 80mm lens, commercial photography, 100% photorealistic, perfect caucasian skin, long hair flowing naturally",
        perfect_franka_composition,
        None, None,
        "City Fashion Free Pose",
        seed=5003
    )
    
    # 4. Street Fashion - No control
    test_fashion_scenario(
        None,  # No control image
        "Urban fashion photoshoot of caucasian franka person with long hair, busy city street setting, natural daylight, lifestyle modeling, street fashion editorial, shot with 50mm lens, environmental portrait, 100% realistic photography, caucasian features, long beautiful hair, confident expression",
        perfect_franka_composition,
        None, None,
        "Street Fashion Editorial",
        seed=5004
    )
    
    # 5. Architectural Fashion - No control
    test_fashion_scenario(
        None,  # No control image
        "High fashion photoshoot of caucasian franka person with long flowing hair, modern architecture background, soft natural lighting, elegant modeling pose, luxury fashion editorial, shot on Sony A7R V with 85mm lens, commercial fashion photography, 100% photorealistic, perfect caucasian skin tone, long gorgeous hair",
        perfect_franka_composition,
        None, None,
        "Architectural Fashion",
        seed=5005
    )
    
    # 6. Cute Sea Creature - Enhanced cuteness
    test_fashion_scenario(
        None,  # No control image
        "Adorable underwater creature with extremely cute round baby face, large expressive doe-like eyes, chubby cheeks, button nose, smooth pale skin with subtle franka person facial features, body proportions of cuddly baby white monkey, long graceful slender bat wings that are translucent with bright glowing bio-luminescent neural networks, intense synaptic lightning patterns in bright pastel colors flowing through wing membranes, swimming upward through crystal clear blue water, volumetric underwater lighting, magical and huggable appearance, National Geographic underwater photography",
        {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",
                "weight": 0.6  # Moderate for cute blend
            }
        },
        None, None,
        "Ultra Cute Sea Creature",
        seed=5006
    )
    
    # 7. Even Cuter Sea Creature
    test_fashion_scenario(
        None,  # No control image
        "Extremely cute and huggable underwater creature with big round baby face, enormous kawaii-style eyes, pudgy cheeks, adorable expression with gentle franka person features, chubby baby monkey body proportions, long elegant slender bat wings with translucent membranes, bright intense bio-luminescent synapses firing with vivid pink, blue, and purple lightning patterns, graceful swimming motion upward, ethereal underwater scene, magical and endearing appearance, professional nature photography",
        {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24", 
                "weight": 0.7  # Slightly stronger for features
            }
        },
        None, None,
        "Kawaii Sea Creature",
        seed=5007
    )
    
    print("\n" + "=" * 80)
    print("✅ Fashion photoshoot and cute creature tests completed!")
    print("📷 All images use the successful seed 4003 settings for maximum Franka likeness")

if __name__ == "__main__":
    run_fashion_photoshoot_tests()