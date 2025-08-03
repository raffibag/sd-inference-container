#!/usr/bin/env python3
"""Photorealistic creature hybrid using real animal anatomy"""

import json
import boto3
import time
import base64
from pathlib import Path

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')

def test_realistic_hybrid():
    """Test photorealistic creature hybrid"""
    
    payload = {
        "prompt": "National Geographic underwater photography of a rare marine mammal, realistic white fur seal pup with delicate membrane wing adaptations similar to flying squirrel gliding membranes, anatomically correct marine mammal with large dark eyes, realistic mammalian facial structure with subtle human-like expressions, translucent wing membranes with visible blood vessels creating natural bio-luminescent patterns, swimming gracefully in crystal clear ocean water, documentary wildlife photography, scientifically accurate anatomy, shot on underwater camera with natural lighting",
        "negative_prompt": "cartoon, anime, fantasy, magical, glowing, artificial, CGI, 3D render, unrealistic proportions, cartoon eyes, stylized, illustration, painting",
        "composition": {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",
                "weight": 0.3  # Very low for subtle facial influence
            }
        },
        "num_images": 1,
        "num_inference_steps": 45,  # More steps for realism
        "guidance_scale": 8.0,  # Higher for prompt adherence
        "width": 1024,
        "height": 1024,
        "seed": 8001
    }
    
    print("🌊 Testing Photorealistic Marine Mammal Hybrid...")
    print(f"📝 Prompt: {payload['prompt']}")
    
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
            filename = f"realistic_hybrid_seal_seed8001.png"
            filepath = outputs_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            print(f"💾 Saved: {filepath}")
            
        else:
            print("❌ No image generated")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_realistic_hybrid_v2():
    """Test another realistic approach - primate with wing adaptations"""
    
    payload = {
        "prompt": "Wildlife documentary photograph of rare albino baby macaque with genetic wing membrane adaptations, realistic primate anatomy with pale white fur, natural primate facial features with subtle human-like expressions reminiscent of franka person, membrane wing structures similar to sugar glider adaptations, anatomically plausible wing membranes with visible vascular patterns, natural forest canopy environment, documentary wildlife photography, National Geographic style, realistic animal anatomy, natural lighting",
        "negative_prompt": "cartoon, anime, fantasy, magical, glowing, artificial, CGI, 3D render, unrealistic proportions, cartoon eyes, stylized, illustration, painting, underwater",
        "composition": {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",
                "weight": 0.4  # Low for subtle influence
            }
        },
        "num_images": 1,
        "num_inference_steps": 45,
        "guidance_scale": 8.0,
        "width": 1024,
        "height": 1024,
        "seed": 8002
    }
    
    print("\n🐒 Testing Photorealistic Primate Wing Hybrid...")
    print(f"📝 Prompt: {payload['prompt']}")
    
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
            filename = f"realistic_hybrid_primate_seed8002.png"
            filepath = outputs_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            print(f"💾 Saved: {filepath}")
            
        else:
            print("❌ No image generated")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def run_realistic_hybrid_tests():
    """Run both realistic hybrid tests"""
    print("🧬 Testing Photorealistic Creature Hybrids...")
    print("=" * 80)
    
    test_realistic_hybrid()
    test_realistic_hybrid_v2()
    
    print("\n" + "=" * 80)
    print("✅ Realistic hybrid tests completed!")
    print("📸 Both should look like real documentary photography")

if __name__ == "__main__":
    run_realistic_hybrid_tests()