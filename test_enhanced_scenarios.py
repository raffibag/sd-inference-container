#!/usr/bin/env python3
"""Enhanced face swaps and detailed fantasy creature with Franka LoRA"""

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

def test_enhanced_scenario(control_image_path, prompt, composition, control_type, conditioning_scale, test_name, seed=42):
    """Test enhanced scenarios with stronger LoRA influence"""
    
    payload = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, bad anatomy, ugly, distorted, artificial, cartoon, CGI, plastic, unrealistic, bad hair, wig-like hair",
        "num_images": 1,
        "num_inference_steps": 40,  # More steps for detail
        "guidance_scale": 6.5,  # Slightly higher for more prompt adherence
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
            filename = f"enhanced_{safe_name}_seed{seed}.png"
            filepath = outputs_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            print(f"💾 Saved: {filepath}")
            
        else:
            print("❌ No image generated")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def run_enhanced_tests():
    """Test enhanced face swaps and detailed fantasy creature"""
    
    print("🎭 Testing Enhanced Face Swaps and Fantasy Creature...")
    print("=" * 80)
    
    control_dir = Path("../_images/control")
    
    # Enhanced face swap composition with stronger weight for hair/features
    strong_franka_composition = {
        "character": {
            "model_path": "auto-2025-01-31-18-05-24",
            "weight": 1.1  # Stronger for more complete face/hair transfer
        }
    }
    
    # 1. Enhanced Face Swap v22 - Emphasize hair and facial features
    test_enhanced_scenario(
        control_dir / "sin_versace_22.jpg",
        "Natural portrait photograph of franka person with her distinctive hair style and facial features, soft natural lighting, realistic skin texture, documentary photography style, natural hair texture and color matching franka person, authentic facial structure, lifestyle photography, minimal makeup",
        strong_franka_composition,
        "openpose",
        0.9,  # Keep successful low conditioning
        "Enhanced Face Swap Hair v22",
        seed=4001
    )
    
    # 2. Enhanced Face Swap v77 - Complete feature replacement
    test_enhanced_scenario(
        control_dir / "sin_versace_77.jpg", 
        "Indoor portrait of franka person showing her complete facial features and hair, soft diffused lighting, natural expression, realistic skin texture, documentary photography style, franka person's distinctive hair and face, minimal makeup, authentic features, lifestyle photography",
        strong_franka_composition,
        "openpose",
        1.0,  # Keep successful conditioning
        "Enhanced Face Swap Complete v77",
        seed=4002
    )
    
    # 3. Ultra Enhanced Face Swap - Maximum feature transfer
    test_enhanced_scenario(
        control_dir / "sin_versace_22.jpg",
        "Professional portrait of franka person, complete facial likeness including hair style, eye color, facial structure, natural lighting, realistic photography, documentary style, authentic franka person features, natural hair texture",
        {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",
                "weight": 1.3  # Maximum weight for strongest resemblance
            }
        },
        "openpose",
        0.8,  # Lower control to let LoRA dominate
        "Ultra Enhanced Face Swap v22",
        seed=4003
    )
    
    # 4. Enhanced Fantasy Creature v1 - Translucent wings with bio-luminescence
    test_enhanced_scenario(
        None,  # No control image
        "Underwater photograph of magical sea creature swimming upwards in clear blue ocean water, creature has translucent bat-like wings with soft pastel bio-luminescent patterns resembling neural networks and synapses firing inside the wing membranes, lightning-like veins of soft pink and blue light, head and body structure like a cute hairless white baby monkey with gentle human facial features inspired by franka person, large expressive eyes, smooth pale skin, graceful swimming motion, ethereal underwater lighting with rays of sunlight filtering through water, magical realism, National Geographic underwater photography",
        {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",
                "weight": 0.7  # Moderate weight for human features
            }
        },
        None, None,
        "Bio Luminescent Sea Creature v1",
        seed=4004
    )
    
    # 5. Enhanced Fantasy Creature v2 - More detailed description
    test_enhanced_scenario(
        None,  # No control image
        "Mystical underwater creature with body proportions of a cute baby white monkey, smooth hairless pale skin, head featuring gentle human facial characteristics from franka person including eye shape and facial structure, large translucent bat wings with intricate bio-luminescent neural network patterns, soft pastel lightning streaks of lavender and cyan flowing through wing membranes like synapses firing, swimming gracefully upward through crystal clear turquoise water, volumetric underwater lighting, coral reef environment below, magical marine biology, ethereal and peaceful expression",
        {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",
                "weight": 0.8  # Slightly stronger for more human features
            }
        },
        None, None,
        "Bio Luminescent Sea Creature v2",
        seed=4005
    )
    
    print("\n" + "=" * 80)
    print("✅ Enhanced scenario tests completed!")
    print("🎭 Check results: stronger face swaps and detailed bio-luminescent creature")

if __name__ == "__main__":
    run_enhanced_tests()