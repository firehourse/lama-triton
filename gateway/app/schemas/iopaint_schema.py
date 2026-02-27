"""Pydantic schemas for inpainting API endpoints."""

from typing import Optional

from pydantic import BaseModel, Field


class InpaintRequest(BaseModel):
    """Standard IOPaint-compatible inpaint request.

    Accepts base64-encoded image and mask. Gateway performs full
    preprocessing (decode, pad, normalize) before forwarding to Triton.
    Drop-in replacement for the original IOPaint ``/api/v1/inpaint``.
    """

    image: str  # base64-encoded image (data URI or raw base64)
    mask: str   # base64-encoded mask (data URI or raw base64)
    ldm_steps: Optional[int] = 20
    ldm_sampler: Optional[str] = "ddim"
    zits_wireframe: Optional[bool] = True
    hd_strategy: Optional[str] = "Crop"
    hd_strategy_crop_margin: Optional[int] = 128
    hd_strategy_crop_trigger_size: Optional[int] = 512
    hd_strategy_resize_limit: Optional[int] = 512
    prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    use_keras_upscale: Optional[bool] = False
    croper_x: Optional[int] = 0
    croper_y: Optional[int] = 0
    croper_height: Optional[int] = 512
    croper_width: Optional[int] = 512
    use_extrapolate: Optional[bool] = False

    class Config:
        extra = "allow"  # accept unknown IOPaint params without error


class PreprocessedInpaintRequest(BaseModel):
    """Pre-processed inpaint request for reduced Gateway CPU usage.

    Caller (e.g. image-toolbox) is responsible for:
    1. Decoding the raw image and mask.
    2. Padding both to a multiple of 8.
    3. Normalizing the image to [0, 1] float32.
    4. Binarizing the mask to {0, 1} float32.
    5. Reshaping to ``[1, 3, H, W]`` and ``[1, 1, H, W]`` respectively.
    6. Serializing via ``numpy_array.tobytes()`` and base64-encoding the result.

    Example (image-toolbox side)::

        import base64
        import numpy as np

        tensor_b64 = base64.b64encode(img_tensor.tobytes()).decode()

    """

    # base64(float32 numpy bytes), shape [1, 3, H_padded, W_padded]
    image_tensor_b64: str = Field(..., description="base64 of float32 numpy bytes, shape [1,3,H,W]")
    # base64(float32 numpy bytes), shape [1, 1, H_padded, W_padded]
    mask_tensor_b64: str = Field(..., description="base64 of float32 numpy bytes, shape [1,1,H,W]")

    # padded tensor shapes (needed to reconstruct numpy arrays from raw bytes)
    image_shape: list[int] = Field(..., description="e.g. [1, 3, 512, 512]")
    mask_shape: list[int] = Field(..., description="e.g. [1, 1, 512, 512]")

    # original dimensions before padding (for post-process cropping)
    original_height: int
    original_width: int
