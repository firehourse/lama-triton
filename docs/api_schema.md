# API Schema Documentation

本專案維持與 IOPaint 100% 相容的 API Schema。以下為標準 JSON Schema 定義。

## 1. Standard Inpaint Request (`/api/v1/inpaint`)

此端點接受原始影像與遮罩的 Base64 字串。

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "InpaintRequest",
  "type": "object",
  "properties": {
    "image": { "type": "string", "description": "Base64-encoded image (data URI or raw base64)" },
    "mask": { "type": "string", "description": "Base64-encoded mask (data URI or raw base64)" },
    "ldm_steps": { "type": "integer", "default": 20 },
    "ldm_sampler": { "type": "string", "default": "ddim" },
    "zits_wireframe": { "type": "boolean", "default": true },
    "hd_strategy": { "type": "string", "default": "Crop", "enum": ["None", "Resize", "Crop"] },
    "hd_strategy_crop_margin": { "type": "integer", "default": 128 },
    "hd_strategy_crop_trigger_size": { "type": "integer", "default": 512 },
    "hd_strategy_resize_limit": { "type": "integer", "default": 512 },
    "prompt": { "type": "string", "default": "" },
    "negative_prompt": { "type": "string", "default": "" },
    "use_keras_upscale": { "type": "boolean", "default": false },
    "croper_x": { "type": "integer", "default": 0 },
    "croper_y": { "type": "integer", "default": 0 },
    "croper_height": { "type": "integer", "default": 512 },
    "croper_width": { "type": "integer", "default": 512 },
    "use_extrapolate": { "type": "boolean", "default": false }
  },
  "required": ["image", "mask"],
  "additionalProperties": true
}
```

## 2. Preprocessed Inpaint Request (`/api/v1/inpaint/preprocessed`)

效能導向端點，傳送原始 Float32 Tensor Bytes 的 Base64 編碼。

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "PreprocessedInpaintRequest",
  "type": "object",
  "properties": {
    "image_tensor_b64": { "type": "string", "description": "Base64 of float32 numpy bytes, shape [1, 3, H, W]" },
    "mask_tensor_b64": { "type": "string", "description": "Base64 of float32 numpy bytes, shape [1, 1, H, W]" },
    "image_shape": { "type": "array", "items": { "type": "integer" }, "minItems": 4, "maxItems": 4 },
    "mask_shape": { "type": "array", "items": { "type": "integer" }, "minItems": 4, "maxItems": 4 },
    "original_height": { "type": "integer" },
    "original_width": { "type": "integer" }
  },
  "required": ["image_tensor_b64", "mask_tensor_b64", "image_shape", "mask_shape", "original_height", "original_width"]
}
```

## Response Format

成功時回傳 `200 OK`，Content-type 為 `text/plain`，內容為結果影像的 Base64 字串。
失敗時回傳 `502 Bad Gateway` 或 `499 Client Closed Request`。

## Response Format

成功時回傳 `200 OK`，Content-type 為 `text/plain`，內容為結果影像的 Base64 字串。
失敗時回傳 `502 Bad Gateway` 或 `499 Client Closed Request`。
