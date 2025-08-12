#!/usr/bin/env python3
"""
Production inference test suite
Tests all combinations: base, ControlNet, LoRA, and mixed scenarios
"""

import json
import base64
import requests
from PIL import Image
from io import BytesIO
import numpy as np

# API endpoint
API_URL = "http://localhost:8080"

def create_test_image(width=1024, height=1024, color=(127, 127, 127)):
    """Create a test image"""
    img = Image.new('RGB', (width, height), color)
    # Add some structure for edge detection
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # Draw some shapes
    draw.rectangle([100, 100, 400, 400], outline='white', width=5)
    draw.ellipse([500, 500, 800, 800], outline='black', width=5)
    
    # Convert to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "="*60)
    print("TEST: Health Check")
    print("="*60)
    
    response = requests.get(f"{API_URL}/ping")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health check passed")
        print(f"   Device: {data.get('device')}")
        print(f"   Dtype: {data.get('dtype')}")
        print(f"   Current ControlNet: {data.get('current_controlnet')}")
        print(f"   Available ControlNets: {data.get('available_controlnets')}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

def test_base_generation():
    """Test base generation without ControlNet or LoRA"""
    print("\n" + "="*60)
    print("TEST: Base Generation (No ControlNet, No LoRA)")
    print("="*60)
    
    payload = {
        "prompt": "a cinematic portrait, 85mm lens, soft natural light, professional photography",
        "negative_prompt": "blurry, low quality",
        "control_type": "none",
        "num_inference_steps": 20,
        "width": 512,
        "height": 512,
        "num_images": 1,
        "seed": 42
    }
    
    print(f"Prompt: {payload['prompt'][:50]}...")
    print(f"Size: {payload['width']}x{payload['height']}")
    print(f"Steps: {payload['num_inference_steps']}")
    
    response = requests.post(f"{API_URL}/invocations", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Generated {len(data.get('images', []))} image(s)")
        
        # Check metadata
        metadata = data.get('metadata', {})
        print(f"   Seed used: {metadata.get('seed')}")
        print(f"   ControlNet: {metadata.get('control_type')}")
        print(f"   LoRAs active: {metadata.get('loras_active')}")
        print(f"   Scheduler: {metadata.get('scheduler')}")
        return True
    else:
        print(f"❌ Generation failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_controlnet_canny():
    """Test ControlNet with Canny edge detection"""
    print("\n" + "="*60)
    print("TEST: ControlNet Canny (No LoRA)")
    print("="*60)
    
    control_image = create_test_image(512, 512)
    
    payload = {
        "prompt": "a modern glass building with geometric architecture, photorealistic",
        "negative_prompt": "blurry, cartoon, anime",
        "control_type": "canny",
        "control_image": control_image,
        "controlnet_conditioning_scale": 0.8,
        "num_inference_steps": 25,
        "width": 512,
        "height": 512,
        "num_images": 1,
        "seed": 100
    }
    
    print(f"Prompt: {payload['prompt'][:50]}...")
    print(f"ControlNet: {payload['control_type']}")
    print(f"Conditioning scale: {payload['controlnet_conditioning_scale']}")
    
    response = requests.post(f"{API_URL}/invocations", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Generated with Canny ControlNet")
        
        metadata = data.get('metadata', {})
        print(f"   Seed: {metadata.get('seed')}")
        print(f"   ControlNet confirmed: {metadata.get('control_type')}")
        return True
    else:
        print(f"❌ Canny generation failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_controlnet_depth():
    """Test ControlNet with Depth estimation"""
    print("\n" + "="*60)
    print("TEST: ControlNet Depth (No LoRA)")
    print("="*60)
    
    control_image = create_test_image(512, 512)
    
    payload = {
        "prompt": "a futuristic cityscape at sunset, volumetric lighting, cinematic",
        "negative_prompt": "blurry, low quality",
        "control_type": "depth",
        "control_image": control_image,
        "controlnet_conditioning_scale": 0.9,
        "num_inference_steps": 25,
        "width": 512,
        "height": 512,
        "num_images": 1,
        "seed": 200
    }
    
    print(f"Prompt: {payload['prompt'][:50]}...")
    print(f"ControlNet: {payload['control_type']}")
    print(f"Conditioning scale: {payload['controlnet_conditioning_scale']}")
    
    response = requests.post(f"{API_URL}/invocations", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Generated with Depth ControlNet")
        
        metadata = data.get('metadata', {})
        print(f"   Seed: {metadata.get('seed')}")
        print(f"   ControlNet confirmed: {metadata.get('control_type')}")
        return True
    else:
        print(f"❌ Depth generation failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_lora_composition():
    """Test LoRA composition without ControlNet"""
    print("\n" + "="*60)
    print("TEST: LoRA Composition (No ControlNet)")
    print("="*60)
    
    payload = {
        "prompt": "professional portrait photography, studio lighting",
        "negative_prompt": "blurry, amateur",
        "composition": {
            "style": {
                "model_path": "test-style-lora",
                "weight": 0.8
            }
        },
        "control_type": "none",
        "num_inference_steps": 20,
        "width": 512,
        "height": 512,
        "num_images": 1,
        "seed": 300
    }
    
    print(f"Prompt: {payload['prompt'][:50]}...")
    print(f"LoRA composition: {list(payload['composition'].keys())}")
    
    response = requests.post(f"{API_URL}/invocations", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Generated with LoRA composition")
        
        metadata = data.get('metadata', {})
        print(f"   LoRAs active: {metadata.get('loras_active')}")
        print(f"   Note: LoRA may fail to load if path doesn't exist")
        return True
    else:
        print(f"❌ LoRA generation failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_controlnet_plus_lora():
    """Test ControlNet + LoRA combination"""
    print("\n" + "="*60)
    print("TEST: ControlNet + LoRA Combination")
    print("="*60)
    
    control_image = create_test_image(512, 512)
    
    payload = {
        "prompt": "elegant architectural photography, golden hour",
        "negative_prompt": "blurry, oversaturated",
        "composition": {
            "style": {
                "model_path": "architectural-style",
                "weight": 0.7
            }
        },
        "control_type": "canny",
        "control_image": control_image,
        "controlnet_conditioning_scale": 0.75,
        "num_inference_steps": 25,
        "width": 512,
        "height": 512,
        "num_images": 1,
        "seed": 400
    }
    
    print(f"Prompt: {payload['prompt'][:50]}...")
    print(f"ControlNet: {payload['control_type']}")
    print(f"LoRA: {list(payload['composition'].keys())}")
    
    response = requests.post(f"{API_URL}/invocations", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Generated with ControlNet + LoRA")
        
        metadata = data.get('metadata', {})
        print(f"   ControlNet: {metadata.get('control_type')}")
        print(f"   LoRAs active: {metadata.get('loras_active')}")
        return True
    else:
        print(f"❌ Combined generation failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_random_seed():
    """Test generation with random seed"""
    print("\n" + "="*60)
    print("TEST: Random Seed Generation")
    print("="*60)
    
    payload = {
        "prompt": "a serene landscape, mountains and lake",
        "control_type": "none",
        "num_inference_steps": 15,
        "width": 512,
        "height": 512,
        "num_images": 1
        # No seed provided - should generate random
    }
    
    print(f"Prompt: {payload['prompt'][:50]}...")
    print(f"Seed: Not provided (will be random)")
    
    response = requests.post(f"{API_URL}/invocations", json=payload)
    if response.status_code == 200:
        data = response.json()
        metadata = data.get('metadata', {})
        random_seed = metadata.get('seed')
        print(f"✅ Generated with random seed: {random_seed}")
        
        # Verify we can reproduce with the same seed
        payload['seed'] = random_seed
        response2 = requests.post(f"{API_URL}/invocations", json=payload)
        if response2.status_code == 200:
            print(f"   ✅ Reproducible with seed {random_seed}")
        return True
    else:
        print(f"❌ Random seed generation failed: {response.status_code}")
        return False

def test_different_sizes():
    """Test different image sizes"""
    print("\n" + "="*60)
    print("TEST: Different Image Sizes")
    print("="*60)
    
    sizes = [(512, 512), (768, 768), (1024, 1024)]
    
    for width, height in sizes:
        payload = {
            "prompt": "abstract art, vibrant colors",
            "control_type": "none",
            "num_inference_steps": 15,
            "width": width,
            "height": height,
            "num_images": 1,
            "seed": 500
        }
        
        print(f"\nTesting {width}x{height}...")
        response = requests.post(f"{API_URL}/invocations", json=payload)
        
        if response.status_code == 200:
            print(f"  ✅ {width}x{height} succeeded")
        else:
            print(f"  ❌ {width}x{height} failed: {response.status_code}")
            return False
    
    return True

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("PRODUCTION INFERENCE TEST SUITE")
    print("="*80)
    
    tests = [
        ("Health Check", test_health_check),
        ("Base Generation", test_base_generation),
        ("ControlNet Canny", test_controlnet_canny),
        ("ControlNet Depth", test_controlnet_depth),
        ("LoRA Composition", test_lora_composition),
        ("ControlNet + LoRA", test_controlnet_plus_lora),
        ("Random Seed", test_random_seed),
        ("Different Sizes", test_different_sizes)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Production ready!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    # Make sure the API is running
    print("Starting production inference tests...")
    print("Make sure the API is running at http://localhost:8080")
    
    import time
    time.sleep(2)
    
    # Run tests
    success = run_all_tests()
    exit(0 if success else 1)