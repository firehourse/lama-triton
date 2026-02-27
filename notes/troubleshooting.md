# Troubleshooting Notes: Inference Backend Architecture

---

## [1] PyTorch JIT to ONNX 轉換失敗 (已解決 — 放棄 ONNX 路線)

### 問題描述
在執行 `torch.onnx.export` 轉換 `big-lama.pt` (TorchScript JIT 格式) 時，連續出現以下錯誤：
- `ValueError: no signature found for builtin`
- `RuntimeError: Failed to convert 'dynamic_axes' to 'dynamic_shapes'`

### 問題根源
1. **TorchScript 內省失敗**: 新版 PyTorch (Dynamo 時代，>= 2.x) 嘗試「解析」預編譯 JIT 模型的內建 C++ 函數簽名時，`inspect.signature` 會失敗。
2. **語法強制遷移**: 新版導出器偏好 `dynamic_shapes`（基於 `torch.export`），但處理舊 JIT 模型時產生邏輯衝突。
3. **LaMa 架構複雜度**: LaMa 使用大量自定義層（FFCResBlock、FourierUnit），這些層在 ONNX opset 14 下難以完整表達，即使強制 `dynamo=False` 也會遇到 graph break。

### 嘗試過的方案
| 方案 | 結果 |
|------|------|
| 方案 A: `dynamo=False` 強制舊路徑 | 仍失敗，signature 問題未解決 |
| 方案 B: 明確給 Input Tensor，跳過 wrapper | 可匯出但 Dynamic Shapes 不穩定 |
| 方案 C: 降版 PyTorch < 2.0 | 依賴衝突，無法與 `tritonclient` 共存 |

### 最終決策：改用 Triton PyTorch Backend (Direct TorchScript)
**放棄 ONNX → TensorRT 路線**，直接讓 Triton 的 `pytorch_libtorch` backend 載入 `big-lama.pt`：

```
weights/big-lama.pt  →  models/lama_pytorch/1/model.pt  →  Triton (pytorch_libtorch)
```

**優點**:
- 零轉換損耗，模型原生載入
- Triton 仍提供 Dynamic Batching + Instance Groups 的並發能力
- 避免 TRT 轉換的精度損失驗證成本

**代價**:
- 失去 TRT 的 FP16、Layer Fusion 等硬體加速（估計 2~3x 速度差距）
- GPU 利用率提升仍透過 Triton batching 實現，非 TRT kernel 融合

### 驗證與生產環境配置
- 部署模組: `lama_pytorch` (Status: READY)
- 引擎後端: `pytorch_libtorch`
- Dynamic Shapes 配置: `[-1, 3, -1, -1]` 
- Instance Groups: `count: 2, KIND_GPU`
- Dynamic Batching 參數: `max_queue_delay: 50ms`

---

