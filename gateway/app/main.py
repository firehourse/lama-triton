"""FastAPI Gateway for lama-triton inference service.

Provides two inpainting endpoints:
- ``POST /api/v1/inpaint``            — IOPaint-compatible; Gateway handles all preprocessing.
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
    pre_process,
)
from gateway.app.schemas.iopaint_schema import InpaintRequest, PreprocessedInpaintRequest

app = FastAPI(title="Lama-Triton Gateway", version="1.0.0")

# Triton configuration — read from environment so Docker Compose can override.
TRITON_URL = os.getenv("TRITON_URL", "localhost:8001")
MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "lama_pytorch")

# Thread-local Triton clients: each worker thread gets its own connection to
# avoid the thread-local context conflicts inside tritonclient.grpc.
_thread_local = threading.local()


def _get_triton_client() -> grpcclient.InferenceServerClient:
    """Return the thread-local Triton gRPC client, creating it if needed."""
    if not hasattr(_thread_local, "client"):
        _thread_local.client = grpcclient.InferenceServerClient(url=TRITON_URL)
    return _thread_local.client


def _blocking_triton_infer(
    img_tensor: np.ndarray,
    mask_tensor: np.ndarray,
) -> np.ndarray:
    """Send a synchronous inference request to Triton.

    This function is intentionally synchronous and designed to be executed
    inside ``asyncio.to_thread`` so it does not block the event loop.

    Args:
        img_tensor:  Float32 array of shape ``[1, 3, H, W]``, values in [0, 1].
        mask_tensor: Float32 array of shape ``[1, 1, H, W]``, values in {0, 1}.

    Returns:
        Float32 output array of shape ``[1, 3, H, W]``.

    """
    client = _get_triton_client()

    inputs = [
        grpcclient.InferInput("image", img_tensor.shape, "FP32"),
        grpcclient.InferInput("mask", mask_tensor.shape, "FP32"),
    ]
    inputs[0].set_data_from_numpy(img_tensor)
    inputs[1].set_data_from_numpy(mask_tensor)

    outputs = [grpcclient.InferRequestedOutput("output")]
    response = client.infer(MODEL_NAME, inputs, outputs=outputs)
    return response.as_numpy("output")


def _encode_result_to_base64(result_bgr: np.ndarray) -> str:
    """Encode a BGR image array to a base64 PNG string."""
    _, buffer = cv2.imencode(".png", result_bgr)
    return base64.b64encode(buffer).decode("utf-8")


async def _run_inference(
    img_tensor: np.ndarray,
    mask_tensor: np.ndarray,
    original_shape: tuple[int, int],
    request: Request,
) -> Response:
    """Shared inference pipeline used by both endpoints.

    Checks for client disconnects before and after the (potentially long)
    Triton call to avoid wasting GPU resources on abandoned requests.

    Args:
        img_tensor:     Preprocessed image tensor ``[1, 3, H, W]``.
        mask_tensor:    Preprocessed mask tensor ``[1, 1, H, W]``.
        original_shape: ``(height, width)`` of the original image before padding.
        request:        FastAPI request object for disconnect detection.

    Returns:
        HTTP response containing base64-encoded PNG result.

    """
    # Check before sending to Triton — saves GPU time if client already left.
    if await request.is_disconnected():
        # 499 = Client Closed Request (nginx convention)
        return Response(status_code=499)

    try:
        output_tensor = await asyncio.to_thread(
            _blocking_triton_infer, img_tensor, mask_tensor
        )
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"error": f"Triton inference failed: {exc}"},
        )

    # Check again — skip encoding if client disconnected during inference.
    if await request.is_disconnected():
        return Response(status_code=499)

    result_bgr = await asyncio.to_thread(post_process, output_tensor, original_shape)
    result_b64 = await asyncio.to_thread(_encode_result_to_base64, result_bgr)

    return Response(content=result_b64, media_type="text/plain")


@app.post("/api/v1/inpaint")
async def inpaint(req: InpaintRequest, request: Request) -> Response:
    """IOPaint-compatible inpainting endpoint.

    Accepts raw base64-encoded image and mask; performs full preprocessing
    (decode → pad to mod-8 → normalize) on the Gateway side before
    forwarding to Triton for inference.

    For lower Gateway CPU usage, prefer ``/api/v1/inpaint/preprocessed``
    and move the preprocessing to the upstream caller (image-toolbox).
    """
    t_start = time.monotonic()

    # Preprocessing is CPU-bound; run in thread pool to avoid blocking
    # the asyncio event loop while other requests are being handled.
    image_np = await asyncio.to_thread(decode_base64_to_image, req.image)
    mask_np = await asyncio.to_thread(decode_base64_to_mask, req.mask)
    original_shape = image_np.shape[:2]

    img_tensor, mask_tensor, _ = await asyncio.to_thread(pre_process, image_np, mask_np)

    response = await _run_inference(img_tensor, mask_tensor, original_shape, request)

    elapsed_ms = (time.monotonic() - t_start) * 1000
    response.headers["X-Inference-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


@app.post("/api/v1/inpaint/preprocessed")
async def inpaint_preprocessed(req: PreprocessedInpaintRequest, request: Request) -> Response:
    """Pre-processed inpainting endpoint for reduced Gateway CPU usage.

    The caller (e.g. image-toolbox) is responsible for all preprocessing:
    decoding, padding to mod-8, normalizing, and binarizing the mask.
    Tensors must be serialized as raw float32 bytes and base64-encoded.

    This endpoint skips all preprocessing on the Gateway side, which
    significantly reduces CPU consumption under high concurrency.

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

    # Decode tensors from base64 bytes; this is lightweight I/O, no blocking concern.
    img_bytes = base64.b64decode(req.image_tensor_b64)
    mask_bytes = base64.b64decode(req.mask_tensor_b64)

    img_tensor = np.frombuffer(
        img_bytes, dtype=np.float32).reshape(req.image_shape)
    mask_tensor = np.frombuffer(
        mask_bytes, dtype=np.float32).reshape(req.mask_shape)

    # Ensure arrays are writable (frombuffer returns read-only views).
    img_tensor = np.ascontiguousarray(img_tensor)
    mask_tensor = np.ascontiguousarray(mask_tensor)

    original_shape = (req.original_height, req.original_width)

    response = await _run_inference(img_tensor, mask_tensor, original_shape, request)

    elapsed_ms = (time.monotonic() - t_start) * 1000
    response.headers["X-Inference-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


@app.get("/health")
async def health() -> dict:
    """Health check endpoint.

    Returns backend service liveness and readiness status. Useful for
    load balancer health probes and ``docker-compose`` ``healthcheck``.
    """
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
    }


@app.get("/")
async def root() -> dict:
    """Service info."""
    return {"service": "lama-triton Gateway", "version": "1.0.0"}
