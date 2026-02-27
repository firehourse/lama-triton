# IOPaint 工作流程分析：為何同時請求會呈現此行為

這份文件主要記錄在對 `IOPaint` 進行壓測與追蹤時，其 FastAPI 路由與 PyTorch 底層互動所產生的不可預期行為（高併發低利用率、P50 延遲暴增）的根本原因分析。

## 1. FastAPI 端點機制 (`api.py`)

在 `IOPaint` 中，API 端點是以普通同步函式定義的：

```python
self.add_api_route("/api/v1/inpaint", self.api_inpaint, methods=["POST"])

def api_inpaint(self, req: InpaintRequest):
    ...
```

因為使用 `def` 而非 `async def`，FastAPI（透過 Starlette）不會阻塞其單執行緒的 asyncio 事件迴圈，而是自動將每個進入的請求分派到 **ThreadPool**（預設最多 40 個同時執行緒）。

**這意味著**：當 6 個 worker 同時命中端點時，FastAPI 會產生 6 個獨立的 Python 執行緒，平行執行 `api_inpaint`。

## 2. PyTorch 與 GIL（全域解釋器鎖）

在 Python 中，GIL 通常會阻止多個執行緒同時執行 Python 位元碼。但當 PyTorch 被要求在 GPU 上執行大量運算（例如矩陣乘法）時，它會 **釋放 GIL**。
因為 GIL 被釋放，這 6 個不同的 Python 執行緒能同時向 GPU 發送 CUDA 執行指令。

### A. 假性併發與執行緒阻塞
PyTorch 使用 **CUDA Streams**。除非另行設定，所有從任意 Python 執行緒發出的 CUDA kernel 都會被推入 **同一個預設 CUDA stream（Stream 0）**。

CUDA stream 會 **順序** 執行操作。因此，即使 6 個 Python 執行緒同時向 GPU 發出指令，GPU 仍會把它們排成一條長隊列：
- Request 1 (Layer 1) → Request 2 (Layer 1) → Request 1 (Layer 2) → Request 3 (Layer 1) → …

在模型推理結束時，每個 Python 執行緒會呼叫 `.cpu().numpy()`，這是一個 **阻塞同步點**——CPU 執行緒會等到其對應的 tensor 完全在 GPU 上計算完畢才繼續。
因為所有計算都排在同一條順序的 CUDA 隊列中，總執行時間大約是 **6 × 單張圖片的執行時間**。所有執行緒在最後的同步點同時取得結果，於是 6 個回應在同一時刻返回。

### B. 低 GPU 計算利用率（30%‑40%）
GPU 之所以設計為大量平行運算，是因為它需要一次處理 **大矩陣**（這就是所謂的 **Batching**）。在 `IOPaint` 的 `model/lama.py` 中，影像會以 `.unsqueeze(0)` 方式處理，**批次大小固定為 1**。
當批次大小為 1 時，矩陣太小，無法點亮 T4 上的所有 CUDA 核心。GPU 被迫以序列方式處理這些小任務（因為它們排在同一個 stream），因此無法達到 100% 的計算利用率，僅停留在 30%‑40% 左右。

### C. 低 GPU 記憶體利用率（10%‑20%）
T4 GPU 具備 16 GB VRAM。單次 LaMa 模型前向傳播（Batch Size = 1）大約會動態分配幾百 MB 的中間激活緩衝。即使 6 個執行緒同時執行，記憶體分配會重疊：
```
6 × 約 300 MB ≈ 1.8 GB
```
再加上模型本身的權重，總使用量正好落在 10%‑20% 的範圍內。

## 4. 架構瓶頸總結

目前的 `IOPaint` 架構在 **Python 層面是意外的並行**，但在 **GPU 層面完全是串行**：
- **不會做 Batching**（批次大小嚴格為 1）。
- 請求在 PyTorch 的預設 CUDA stream 中依序排隊。
- GPU 有足夠的記憶體與計算資源，但因為軟體架構只給它送出極小的 batch=1 矩陣，導致無法發揮真正的平行處理能力。
- 缺乏 **動態合批 (Dynamic Batching)** 機制，無法將零散併發的 `[1, C, H, W]` 請求重組為 `[N, C, H, W]` 的大矩陣計算。
- 硬體閒置率高，計算密集度不足以喚醒 GPU 高效能運算狀態。此為傳統同步/多線程 Web 框架直接封裝 AI 推理代碼時常見的架構設計缺陷。
