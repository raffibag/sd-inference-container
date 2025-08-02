import os
import logging
import torch
from diffusers import ControlNetModel
from controlnet_aux import (
    CannyDetector,
    OpenposeDetector, 
    MidasDetector,
    MLSDdetector,
    HEDdetector,
    LineartDetector
)

logger = logging.getLogger(__name__)

def register_controlnets(pipe, controlnet_config):
    """
    Registers ControlNet models and their associated preprocessors into the pipeline.

    Args:
        pipe: An initialized `StableDiffusionControlNetPipeline` or similar
        controlnet_config (dict): Dictionary with structure:
            {
                "canny": "diffusers/controlnet-canny-sdxl-1.0-small",
                "depth": "diffusers/controlnet-depth-sdxl-1.0-small", 
                ...
            }

    Returns:
        Tuple of (controlnets dict, controlnet_processors dict)
    """
    controlnets = {}
    controlnet_processors = {}
    
    # Get device and dtype from pipeline
    device = pipe.device
    dtype = pipe.dtype
    
    # Get HuggingFace token
    hf_token = os.environ.get('HUGGINGFACE_HUB_TOKEN')

    processor_map = {
        "canny": lambda: CannyDetector(),
        "depth": lambda: MidasDetector.from_pretrained("lllyasviel/Annotators"),
        "openpose": lambda: OpenposeDetector.from_pretrained("lllyasviel/Annotators"),
        "mlsd": lambda: MLSDdetector(),
        "hed": lambda: HEDdetector(),
        "lineart": lambda: LineartDetector.from_pretrained("lllyasviel/lineart"),
        # Note: ScribbleDetector not available in controlnet_aux
        # Can use HED or PidiNet as alternatives for edge detection
    }

    for control_type, model_path in controlnet_config.items():
        # Load ControlNet model
        try:
            logger.info(f"🔄 Loading ControlNet model for '{control_type}' from {model_path}")
            controlnets[control_type] = ControlNetModel.from_pretrained(
                model_path,
                torch_dtype=dtype,
                use_auth_token=hf_token
            ).to(device)
            logger.info(f"✅ Loaded ControlNet model '{control_type}'")
        except Exception as e:
            logger.error(f"❌ Failed to load ControlNet model '{control_type}': {e}")

        # Load processor
        if control_type in processor_map:
            try:
                logger.info(f"🔄 Initializing processor for '{control_type}'")
                controlnet_processors[control_type] = processor_map[control_type]()
                logger.info(f"✅ Initialized processor '{control_type}'")
            except Exception as e:
                logger.error(f"❌ Failed to initialize preprocessor for '{control_type}': {e}")
        else:
            logger.warning(f"⚠️ No known preprocessor for '{control_type}'")

    # Inject into pipeline
    pipe.controlnets = controlnets
    logger.info(f"✅ Registered {len(controlnets)} ControlNets and {len(controlnet_processors)} processors")
    return controlnets, controlnet_processors