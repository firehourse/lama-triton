"""Image preprocessing and postprocessing utilities for LaMa inpainting.

All functions are synchronous and CPU-bound (numpy operations). Callers
inside ``async def`` endpoints should invoke them via ``asyncio.to_thread``
to avoid blocking the event loop.
"""

import base64
import io

import cv2
import numpy as np
from PIL import Image, ImageOps


def decode_base64_to_image(encoding: str) -> np.ndarray:
    """Decode a base64-encoded image string to an RGB numpy array.

    Accepts both raw base64 and data-URI formats (``data:image/...;base64,...``).
    EXIF orientation is applied automatically. RGBA images are converted to RGB.

    Args:
        encoding: Base64 string or data URI representing the image.

    Returns:
        ``np.ndarray`` of shape ``[H, W, 3]`` with dtype ``uint8``, RGB channel order.

    """
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]

    image_bytes = base64.b64decode(encoding)
    image = Image.open(io.BytesIO(image_bytes))

    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass  # non-critical; skip if EXIF data is missing or corrupt

    if image.mode == "RGBA":
        np_img = np.array(image)
        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
    else:
        image = image.convert("RGB")
        np_img = np.array(image)

    return np_img


def decode_base64_to_mask(encoding: str) -> np.ndarray:
    """Decode a base64-encoded mask string to a grayscale numpy array.

    Args:
        encoding: Base64 string or data URI representing the mask image.

    Returns:
        ``np.ndarray`` of shape ``[H, W]`` with dtype ``uint8``, values in [0, 255].

    """
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]

    mask_bytes = base64.b64decode(encoding)
    mask = Image.open(io.BytesIO(mask_bytes))
    mask = mask.convert("L")
    return np.array(mask)


def _ceil_modulo(x: int, mod: int) -> int:
    """Return the smallest multiple of ``mod`` that is >= ``x``."""
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def _pad_to_modulo(arr: np.ndarray, mod: int) -> np.ndarray:
    """Symmetrically pad ``arr`` so both spatial dimensions are multiples of ``mod``.

    Args:
        arr: Array of shape ``[H, W]`` or ``[H, W, C]``.
        mod: Target modulus (e.g. 8 for LaMa).

    Returns:
        Padded array with the same dtype as ``arr``.

    """
    h, w = arr.shape[:2]
    out_h = _ceil_modulo(h, mod)
    out_w = _ceil_modulo(w, mod)

    pad_h = out_h - h
    pad_w = out_w - w

    if arr.ndim == 3:
        padding = ((0, pad_h), (0, pad_w), (0, 0))
    else:
        padding = ((0, pad_h), (0, pad_w))

    return np.pad(arr, padding, mode="symmetric")


def pre_process(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Prepare raw image and mask arrays for LaMa inference.

    Steps:
    1. Pad both arrays so H and W are multiples of 8 (LaMa requirement).
    2. Normalize image pixels from [0, 255] to [0, 1].
    3. Reshape image to ``[1, 3, H, W]`` (NCHW, channel-first).
    4. Reshape mask to ``[1, 1, H, W]`` and binarize to {0, 1}.

    Args:
        image: RGB image array of shape ``[H, W, 3]``, dtype ``uint8``.
        mask:  Grayscale mask array of shape ``[H, W]``, dtype ``uint8``.

    Returns:
        Tuple of:
        - ``img_tensor``:  Float32 array ``[1, 3, H_pad, W_pad]``, values in [0, 1].
        - ``mask_tensor``: Float32 array ``[1, 1, H_pad, W_pad]``, values in {0, 1}.
        - ``padded_shape``: ``(H_pad, W_pad)`` used for cropping in post-processing.

    """
    padded_image = _pad_to_modulo(image, 8)
    padded_mask = _pad_to_modulo(mask, 8)

    # [H, W, 3] → [1, 3, H, W], normalize to [0, 1]
    img_tensor = padded_image.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)

    # [H, W] → [1, 1, H, W], binary {0, 1}
    mask_tensor = padded_mask.astype(np.float32) / 255.0
    mask_tensor = np.expand_dims(mask_tensor, axis=0)  # [1, H, W]
    mask_tensor = np.expand_dims(mask_tensor, axis=0)  # [1, 1, H, W]
    mask_tensor = (mask_tensor > 0).astype(np.float32)

    return img_tensor, mask_tensor, padded_image.shape[:2]


def post_process(output_tensor: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    """Convert Triton output tensor back to a displayable BGR image.

    Args:
        output_tensor:  Float32 array of shape ``[1, 3, H_pad, W_pad]``, values in [0, 1].
        original_shape: ``(H, W)`` of the original image before padding.

    Returns:
        ``np.ndarray`` of shape ``[H, W, 3]``, dtype ``uint8``, BGR channel order
        (compatible with ``cv2.imwrite`` and the IOPaint response convention).

    """
    # [1, 3, H, W] → [H, W, 3]
    res = output_tensor[0].transpose(1, 2, 0)
    res = np.clip(res * 255, 0, 255).astype(np.uint8)

    # Crop back to original dimensions (remove padding).
    h, w = original_shape
    res = res[:h, :w, :]

    # Convert RGB → BGR for OpenCV / IOPaint compatibility.
    return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
