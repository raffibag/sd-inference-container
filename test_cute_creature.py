#!/usr/bin/env python3
"""Cute sea creature with enhanced bio-luminescence"""

import json
import boto3
import time
import base64
from pathlib import Path

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')

def test_cute_creature():
    """Test ultra cute sea creature with bright bio-luminescent wings"""
    
    payload = {
        "prompt": "Extremely cute and huggable underwater creature with big round baby face, enormous kawaii-style doe eyes, pudgy cheeks, adorable button nose, gentle smile with subtle franka person facial features, chubby baby white monkey body proportions, long elegant slender bat wings with translucent membranes, bright intense bio-luminescent synapses firing with vivid electric pink, cyan blue, and purple lightning patterns coursing through wing membranes, graceful swimming motion upward through crystal clear turquoise water, volumetric underwater lighting with sun rays, magical and endearing appearance, National Geographic underwater photography, extremely cute and huggable",
        "negative_prompt": "scary, ugly, realistic human, adult, mature, dark, threatening, angular features, sharp teeth, large body, thick wings, dim lighting, murky water",
        "composition": {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",
                "weight": 0.7  # Moderate for cute blend with human features
            }
        },
        "num_images": 1,
        "num_inference_steps": 40,
        "guidance_scale": 7.0,  # Higher for more prompt adherence
        "width": 1024,
        "height": 1024,
        "seed": 7001
    }
    
    print("🧜‍♀️ Testing Ultra Cute Bio-Luminescent Sea Creature...")
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
            filename = f"cute_sea_creature_seed7001.png"
            filepath = outputs_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            print(f"💾 Saved: {filepath}")
            
        else:
            print("❌ No image generated")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_cute_creature()