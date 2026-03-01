"""FastAPI Gateway for lama-triton inference service.

Provides two inpainting endpoints:
- ``POST /api/v1/inpaint``            — IOPaint-compatible; C++ Triton Backend handles all preprocessing.
- ``POST /api/v1/inpaint/preprocessed`` — caller supplies pre-processed tensors; reduces Gateway CPU load.
"""

import asyncio
import base64
import os
import time
import threading

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from gateway.app.core.image_processing import (
    decode_base64_to_image,
    decode_base64_to_mask,
    post_process,
)
from gateway.app.schemas.iopaint_schema import InpaintRequest, PreprocessedInpaintRequest

app = FastAPI(title="Lama-Triton Gateway", version="2.0.0")

# Triton configuration — read from environment so Docker Compose can override.
TRITON_URL = os.getenv("TRITON_URL", "localhost:8001")

# v2: Default to ensemble model. C++ backend handles preprocessing.
# Set TRITON_MODEL_NAME=lama_pytorch to fall back to Python-side preprocessing.
MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "lama_ensemble")
USE_ENSEMBLE = os.getenv("USE_ENSEMBLE", "true").lower() == "true"

# Thread-local Triton clients: each worker thread gets its own connection to
# avoid the thread-local context conflicts inside tritonclient.grpc.
_thread_local = threading.local()


def _get_triton_client() -> grpcclient.InferenceServerClient:
    """Return the thread-local Triton gRPC client, creating it if needed."""
    if not hasattr(_thread_local, "client"):
        _thread_local.client = grpcclient.InferenceServerClient(url=TRITON_URL)
    return _thread_local.client


def _blocking_triton_infer_ensemble(
    img_raw: bytes,
    mask_raw: bytes,
) -> tuple[np.ndarray, np.ndarray]:
    """Send raw image/mask bytes to Triton Ensemble (C++ preprocessing).

    Returns:
        tuple: (output_image_tensor, original_shape_tensor)
    """
    client = _get_triton_client()

    img_input = grpcclient.InferInput("IMAGE_RAW", [1], "BYTES")
    mask_input = grpcclient.InferInput("MASK_RAW", [1], "BYTES")
    img_input.set_data_from_numpy(np.array([img_raw], dtype=object))
    mask_input.set_data_from_numpy(np.array([mask_raw], dtype=object))

    outputs = [
        grpcclient.InferRequestedOutput("OUTPUT_IMAGE"),
        grpcclient.InferRequestedOutput("ORIGINAL_SHAPE"),
    ]
    response = client.infer(
        MODEL_NAME, [img_input, mask_input], outputs=outputs)
    return response.as_numpy("OUTPUT_IMAGE"), response.as_numpy("ORIGINAL_SHAPE")


def _blocking_triton_infer_tensors(
    img_tensor: np.ndarray,
    mask_tensor: np.ndarray,
) -> np.ndarray:
    """Send pre-processed tensors to Triton (legacy / fallback path).

    Args:
        img_tensor:  Float32 array of shape ``[1, 3, H, W]``, values in [0, 1].
        mask_tensor: Float32 array of shape ``[1, 1, H, W]``, values in {0, 1}.

    Returns:
        Float32 output array of shape ``[1, 3, H, W]``.
    """
    client = _get_triton_client()

    inputs = [
        grpcclient.InferInput("image", img_tensor.shape, "FP32"),
        grpcclient.InferInput("mask",  mask_tensor.shape, "FP32"),
    ]
    inputs[0].set_data_from_numpy(img_tensor)
    inputs[1].set_data_from_numpy(mask_tensor)

    outputs = [grpcclient.InferRequestedOutput("output")]
    response = client.infer("lama_pytorch", inputs, outputs=outputs)
    return response.as_numpy("output")


def _encode_result_to_base64(result_bgr: np.ndarray) -> str:
    """Encode a BGR image array to a base64 PNG string."""
    _, buffer = cv2.imencode(".png", result_bgr)
    return base64.b64encode(buffer).decode("utf-8")


async def _run_inference_ensemble(
    img_raw: bytes,
    mask_raw: bytes,
    request: Request,
) -> Response:
    """Inference pipeline using Triton Ensemble (C++ preprocessing path)."""
    if await request.is_disconnected():
        return Response(status_code=499)

    try:
        output_tensor, shape_tensor = await asyncio.to_thread(
            _blocking_triton_infer_ensemble, img_raw, mask_raw
        )
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"error": f"Triton ensemble inference failed: {exc}"},
        )

    if await request.is_disconnected():
        return Response(status_code=499)

    # shape_tensor is returned as [H, W] from the C++ backend
    original_shape = (int(shape_tensor[0]), int(shape_tensor[1]))

    # max_batch_size=0 means Triton returns [3,H,W] not [1,3,H,W]; add batch dim
    if output_tensor.ndim == 3:
        output_tensor = output_tensor[np.newaxis, ...]

    result_bgr = await asyncio.to_thread(post_process, output_tensor, original_shape)
    result_b64 = await asyncio.to_thread(_encode_result_to_base64, result_bgr)
    return Response(content=result_b64, media_type="text/plain")


async def _run_inference_tensors(
    img_tensor: np.ndarray,
    mask_tensor: np.ndarray,
    original_shape: tuple[int, int],
    request: Request,
) -> Response:
    """Inference pipeline using pre-processed tensors (legacy Python path)."""
    if await request.is_disconnected():
        return Response(status_code=499)

    try:
        output_tensor = await asyncio.to_thread(
            _blocking_triton_infer_tensors, img_tensor, mask_tensor
        )
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"error": f"Triton inference failed: {exc}"},
        )

    if await request.is_disconnected():
        return Response(status_code=499)

    result_bgr = await asyncio.to_thread(post_process, output_tensor, original_shape)
    result_b64 = await asyncio.to_thread(_encode_result_to_base64, result_bgr)
    return Response(content=result_b64, media_type="text/plain")


@app.post("/api/v1/inpaint")
async def inpaint(req: InpaintRequest, request: Request) -> Response:
    """IOPaint-compatible inpainting endpoint.

    v2 (default): Forwards raw bytes to Triton Ensemble. C++ backend handles
    all preprocessing. Eliminates Python-side CPU cost.
    """
    t_start = time.monotonic()

    if USE_ENSEMBLE:
        # Avoid redundant decoding in the Python Gateway.
        # Decode base64 string → raw bytes and let C++ backend do the rest!
        img_raw = base64.b64decode(
            req.image.split(",")[-1] if "," in req.image else req.image
        )
        mask_raw = base64.b64decode(
            req.mask.split(",")[-1] if "," in req.mask else req.mask
        )
        response = await _run_inference_ensemble(img_raw, mask_raw, request)
    else:
        # Legacy Python preprocessing path (Fallback).
        image_np = await asyncio.to_thread(decode_base64_to_image, req.image)
        mask_np = await asyncio.to_thread(decode_base64_to_mask, req.mask)
        original_shape = image_np.shape[:2]

        from gateway.app.core.image_processing import pre_process
        img_tensor, mask_tensor, _ = await asyncio.to_thread(pre_process, image_np, mask_np)
        response = await _run_inference_tensors(img_tensor, mask_tensor, original_shape, request)

    elapsed_ms = (time.monotonic() - t_start) * 1000
    response.headers["X-Inference-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


@app.post("/api/v1/inpaint/preprocessed")
async def inpaint_preprocessed(req: PreprocessedInpaintRequest, request: Request) -> Response:
    """Pre-processed inpainting endpoint for reduced Gateway CPU usage.

    The caller (e.g. image-toolbox) is responsible for all preprocessing:
    decoding, padding to mod-8, normalizing, and binarizing the mask.
    Tensors must be serialized as raw float32 bytes and base64-encoded.

    Tensor format expected from caller::

        import base64, numpy as np
        # img_tensor shape: [1, 3, H_padded, W_padded], dtype=float32
        payload = {
            "image_tensor_b64": base64.b64encode(img_tensor.tobytes()).decode(),
            "mask_tensor_b64":  base64.b64encode(mask_tensor.tobytes()).decode(),
            "image_shape": list(img_tensor.shape),   # [1, 3, H, W]
            "mask_shape":  list(mask_tensor.shape),  # [1, 1, H, W]
            "original_height": orig_h,
            "original_width":  orig_w,
        }

    """
    t_start = time.monotonic()

    img_bytes = base64.b64decode(req.image_tensor_b64)
    mask_bytes = base64.b64decode(req.mask_tensor_b64)

    img_tensor = np.frombuffer(
        img_bytes, dtype=np.float32).reshape(req.image_shape)
    mask_tensor = np.frombuffer(
        mask_bytes, dtype=np.float32).reshape(req.mask_shape)

    img_tensor = np.ascontiguousarray(img_tensor)
    mask_tensor = np.ascontiguousarray(mask_tensor)

    original_shape = (req.original_height, req.original_width)

    response = await _run_inference_tensors(img_tensor, mask_tensor, original_shape, request)

    elapsed_ms = (time.monotonic() - t_start) * 1000
    response.headers["X-Inference-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    try:
        client = _get_triton_client()
        live = client.is_server_live()
        ready = client.is_server_ready()
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": str(exc)},
        )
    return {
        "status": "ok" if (live and ready) else "degraded",
        "backend_live": live,
        "backend_ready": ready,
        "mode": "ensemble_cpp" if USE_ENSEMBLE else "python_preprocess",
    }


@app.get("/")
async def root() -> dict:
    """Service info."""
    return {
        "service": "lama-triton Gateway",
        "version": "2.0.0",
        "mode": "ensemble_cpp" if USE_ENSEMBLE else "python_preprocess",
    }
