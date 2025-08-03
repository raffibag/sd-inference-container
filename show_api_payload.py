#!/usr/bin/env python3
"""Shows the exact API payload structure for ControlNet + LoRA inference"""

import json
import base64
from pathlib import Path

def show_controlnet_lora_payload():
    """Display the complete API payload structure"""
    
    # Example payload with all parameters
    payload = {
        # Main prompt
        "prompt": "Professional fashion portrait of franka person in elegant black dress, dramatic studio lighting",
        
        # Negative prompt
        "negative_prompt": "blurry, low quality, bad anatomy, ugly, distorted",
        
        # LoRA composition - character and/or style
        "composition": {
            "character": {
                "model_path": "auto-2025-01-31-18-05-24",  # Franka character LoRA
                "weight": 1.0  # Weight 0.0-2.0, typically 0.8-1.3
            },
            "style": {
                "model_path": "auto-2025-01-31-17-52-39",  # Sinisha style LoRA
                "weight": 0.8  # Weight 0.0-2.0, typically 0.6-1.0
            }
        },
        
        # Control image - base64 encoded
        "control_image": "<base64_encoded_image_data_here>",  # From ../_images/control/sin_versace_22.jpg
        
        # ControlNet type
        "control_type": "openpose",  # Options: "canny", "openpose", "depth"
        
        # ControlNet strength
        "controlnet_conditioning_scale": 1.0,  # 0.0-2.0, higher = stronger pose control
        
        # Generation parameters
        "num_images": 1,
        "num_inference_steps": 35,  # 20-50, more = better quality but slower
        "guidance_scale": 7.5,      # 1-20, higher = more prompt adherence
        "width": 1024,
        "height": 1024,
        "seed": 42  # For reproducibility
    }
    
    print("📋 COMPLETE API PAYLOAD STRUCTURE:")
    print("=" * 80)
    print(json.dumps(payload, indent=2))
    print("=" * 80)
    
    print("\n🔍 PARAMETER EXPLANATIONS:")
    print("- prompt: Your main text description")
    print("- composition.character.model_path: 'auto-2025-01-31-18-05-24' (Franka)")
    print("- composition.style.model_path: 'auto-2025-01-31-17-52-39' (Sinisha)")
    print("- control_image: Base64 encoded image from ../_images/control/")
    print("- control_type: 'canny' (edges), 'openpose' (pose), 'depth' (3D)")
    print("- controlnet_conditioning_scale: 0.8-1.5 typical (strength of control)")
    
    print("\n📁 CONTROL IMAGE PATHS:")
    control_dir = Path("../_images/control")
    for img in control_dir.glob("*.jpg"):
        print(f"- {img}")
    
    print("\n🚀 ENDPOINT URL:")
    print("POST https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/vibez-inference-endpoint/invocations")
    
    print("\n🔐 HEADERS:")
    print("Content-Type: application/json")
    print("Authorization: AWS4-HMAC-SHA256 (handled by boto3)")

def show_minimal_payload():
    """Show minimal payload without LoRA or ControlNet"""
    
    minimal = {
        "prompt": "A beautiful landscape at sunset",
        "num_images": 1,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "width": 1024,
        "height": 1024
    }
    
    print("\n\n📋 MINIMAL PAYLOAD (No LoRA, No ControlNet):")
    print("=" * 80)
    print(json.dumps(minimal, indent=2))

def show_example_api_call():
    """Show example of how the actual API call is made"""
    
    print("\n\n🐍 PYTHON CODE TO MAKE THE API CALL:")
    print("=" * 80)
    print("""
import boto3
import json
import base64

# Initialize client
runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')

# Read and encode control image
with open('../_images/control/sin_versace_22.jpg', 'rb') as f:
    control_image_b64 = base64.b64encode(f.read()).decode('utf-8')

# Build payload
payload = {
    "prompt": "Your custom prompt here",
    "composition": {
        "character": {
            "model_path": "auto-2025-01-31-18-05-24",
            "weight": 1.0
        }
    },
    "control_image": control_image_b64,
    "control_type": "openpose",
    "controlnet_conditioning_scale": 1.0,
    "num_images": 1,
    "num_inference_steps": 35,
    "guidance_scale": 7.5,
    "width": 1024,
    "height": 1024,
    "seed": 42
}

# Make API call
response = runtime.invoke_endpoint(
    EndpointName='vibez-inference-endpoint',
    ContentType='application/json',
    Body=json.dumps(payload)
)

# Parse response
result = json.loads(response['Body'].read())
images = result['images']  # List of base64 encoded images
""")

if __name__ == "__main__":
    show_controlnet_lora_payload()
    show_minimal_payload()
    show_example_api_call()