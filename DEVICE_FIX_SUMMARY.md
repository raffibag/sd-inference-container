# ControlNet Device/Dtype Fix Summary

## Problem
The error "Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!" occurred when using depth ControlNet because:
1. ControlNets were loaded with inconsistent dtype (float32 vs float16)
2. ControlNets weren't moved to the correct device (GPU)
3. When switching ControlNets, the new model stayed on CPU
4. Generators defaulted to CPU device

## Solution Implementation

### 1. Single Source of Truth (Lines 44-50 in inference_fixed.py)
```python
# BEFORE: Device and dtype determined in multiple places
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# AFTER: Single source of truth
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
```

### 2. Consistent ControlNet Loading (Lines 90-104)
```python
# BEFORE: ControlNets loaded without device placement
controlnet = ControlNetModel.from_pretrained(
    model_id,
    torch_dtype=torch.float32  # Hardcoded dtype
)
self.controlnets[name] = controlnet  # Not moved to device

# AFTER: Consistent loading with device placement
def load_controlnet(repo_id: str, name: str = None) -> ControlNetModel:
    cn = ControlNetModel.from_pretrained(
        repo_id,
        torch_dtype=DTYPE,  # Use global dtype
        use_safetensors=True
    )
    cn = cn.to(DEVICE)  # Always move to device
    return cn
```

### 3. Safe ControlNet Switching (Lines 195-213)
```python
# BEFORE: Direct assignment without device check
self.pipeline.controlnet = current_controlnet

# AFTER: Ensure device/dtype consistency when switching
def switch_controlnet(self, control_type: ControlNetType):
    new_cn = self.controlnets[control_type.value]
    # Ensure the new ControlNet is on the same device/dtype
    new_cn = new_cn.to(device=DEVICE, dtype=DTYPE)
    self.pipeline.controlnet = new_cn
    # Verify consistency after switch
    self._verify_device_consistency()
```

### 4. Control Image Processing (Lines 215-275)
```python
# BEFORE: Manual tensor conversion could create CPU tensors
depth_tensor = torch.from_numpy(depth_array)  # Default device is CPU

# AFTER: Return PIL/numpy and let diffusers handle conversion
def process_control_image(self, image_data: str, control_type: ControlNetType):
    # ... process image ...
    return processed  # Return PIL or numpy, NOT tensor
    
# In generation:
images = self.pipeline(
    image=control_image,  # Pass PIL/numpy, diffusers handles device placement
    ...
)
```

### 5. Generator on Correct Device (Line 385)
```python
# BEFORE: Generator defaults to CPU
generator = torch.Generator().manual_seed(seed)  # CPU!

# AFTER: Generator on same device as pipeline
generator = torch.Generator(device=DEVICE).manual_seed(seed)
```

### 6. Device Consistency Verification (Lines 277-299)
```python
def _verify_device_consistency(self):
    """Verify all components are on the same device with same dtype"""
    unet_device = next(self.pipeline.unet.parameters()).device
    cn_device = next(self.pipeline.controlnet.parameters()).device
    
    assert unet_device == cn_device, f"Device mismatch: UNet {unet_device} vs ControlNet {cn_device}"
    # ... additional checks ...
```

## Key Gotchas Addressed

1. **float16 on CPU crashes** - Only use float16 when CUDA is available
2. **Generator defaults to CPU** - Always specify device explicitly
3. **Depth processor device** - Specify device when loading transformers pipeline
4. **PIL vs Tensor** - Let diffusers handle conversion to avoid device issues
5. **Runtime switching** - Always re-apply device/dtype when switching models

## Testing

Run the test script to verify device consistency:
```bash
python test_device_consistency.py
```

## Deployment

1. Replace `inference.py` with `inference_fixed.py` in the Docker container
2. Rebuild container: `docker build -t sd-inference:fixed .`
3. Push to ECR: `docker push ...`
4. Redeploy SageMaker endpoint with new container

## Result

With these fixes:
- All tensors guaranteed to be on the same device
- Consistent dtype throughout the pipeline
- Depth ControlNet (and all other ControlNets) work without device errors
- Runtime ControlNet switching is safe
- Better performance with float16 on GPU