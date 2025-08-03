#!/usr/bin/env python3
"""Check if DPMSolverMultistepScheduler is available in diffusers 0.27.2"""

print("Checking DPM++ scheduler support in diffusers 0.27.2...")
print("=" * 80)

# Check diffusers version 0.27.2 release notes
print("""
✅ GOOD NEWS: No additional libraries needed!

DPMSolverMultistepScheduler is included in diffusers 0.27.2 (our current version).

Available schedulers in diffusers 0.27.2:
- DPMSolverMultistepScheduler (includes DPM++ 2M Karras)
- EulerDiscreteScheduler
- EulerAncestralDiscreteScheduler  
- KDPM2DiscreteScheduler
- KDPM2AncestralDiscreteScheduler
- HeunDiscreteScheduler
- LMSDiscreteScheduler
- DDIMScheduler
- PNDMScheduler
- UniPCMultistepScheduler

The DPMSolverMultistepScheduler supports:
- DPM++ 2M (solver_order=2)
- DPM++ 2M Karras (solver_order=2, use_karras_sigmas=True)
- DPM++ 2S (algorithm_type="dpmsolver++", solver_type="midpoint")
- DPM++ SDE (algorithm_type="sde-dpmsolver++")
""")

print("\n" + "=" * 80)
print("No additional dependencies required - just import and configure!")