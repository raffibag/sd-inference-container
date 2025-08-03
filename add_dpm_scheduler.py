#!/usr/bin/env python3
"""
Shows how to add DPM++ 2M Karras scheduler to the inference handler
"""

print("To add DPM++ 2M Karras scheduler support, add this to controlnet_lora_handler.py:")
print("=" * 80)
print("""
# In the imports section, add:
from diffusers import DPMSolverMultistepScheduler

# After loading the pipelines (around line 171), add:

# Configure DPM++ 2M Karras scheduler for both pipelines
logger.info("🎛️  Setting up DPM++ 2M Karras scheduler...")
scheduler_config = {
    "algorithm_type": "dpmsolver++",
    "solver_order": 2,
    "use_karras_sigmas": True,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
}

# Apply to regular pipeline
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    **scheduler_config
)

# Apply to ControlNet pipeline  
controlnet_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    controlnet_pipe.scheduler.config,
    **scheduler_config
)

logger.info("✅ DPM++ 2M Karras scheduler configured")
""")

print("\n" + "=" * 80)
print("This would set both pipelines to use DPM++ 2M Karras by default.")
print("\nAlternatively, you could make it configurable via the API payload:")
print("""
# In the payload:
{
    "prompt": "...",
    "scheduler": "dpm++_2m_karras",  # or "euler_a", "pndm", etc.
    ...
}
""")