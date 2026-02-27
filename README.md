# High-Performance Image Inpainting System

本系統基於異步網關與原生 C++ 推理引擎構建的高效能影像修補 (Inpainting) 服務。專為解決高並發請求下，使用 `IOPaint` 時常見的 GPU 利用率瓶頸與任務排隊積壓問題而設計。

## 快速啟動

### 1. 啟動服務
本專案已整合整合自動化權重下載機制與容器編排。

```bash
docker-compose up --build -d
```

- **API Gateway**: http://localhost:8090
- **Inference Engine Backend**: http://localhost:8001 (gRPC) / 8000 (HTTP)

### 2. 驗證服務
```bash
# 檢查模型就緒情況
curl http://localhost:8000/v2/models/lama_pytorch
```

---

## 系統架構與設計

### 核心組件職責
- **Inference Gateway**: 系統接入與前處理層。採用非同步 (Asynchronous I/O) 設計，具備「連線中斷監控」機制，防止客戶端異常斷線後持續佔用 GPU 算力。
- **Inference Engine**: 底層 C++ 高效能推理引擎。透過 **Dynamic Batching** 將來自不同 Worker 的請求於極短的微秒級時間窗內進行矩陣合併 (Tensor Concat)，極大化 GPU 內核吞吐量。
- **Post-processing**: 將張量還原、裁切回原始解析度，並進行通道轉換。

詳細設計說明請參閱: [architecture.md](./architecture.md)

---

## API 端點說明

### 影像修補 (Inpainting)

| Method | Endpoint                    | Description                                      |
| :----- | :-------------------------- | :----------------------------------------------- |
| POST   | /api/v1/inpaint             | 100% 兼容 `IOPaint` 格式，Gateway 負責完整的影像解碼、填充與前處理 |
| POST   | /api/v1/inpaint/preprocessed| 進階端點。接受序列化 Tensor Bytes，跳過 Gateway 前處理以轉移 CPU 開銷 |

---

## 技術亮點與實作細節

1. **Dynamic Batching (動態合批)**:
   自動收集微秒級時間窗內的請求並組成大矩陣執行。在 6 併發條件下，P50 延遲與 `IOPaint` 相比降低約 60% 以上，整體吞吐量提升 2.5 倍。

2. **全非同步與微服務解耦**:
   Gateway 僅負責排程與數據搬移，模型權重與推論負載完全由獨立 Engine 承載。單節點 Gateway 可處理大量併發不會發生 Thread Pool Starvation。

3. **效能監控與優化**:
   - **CPU 優化**: 將影像解碼與前處理集中處理，整機 CPU 消耗較 `IOPaint` 降低 **75%**。
   - **Ghost Task Prevention**: 偵測 HTTP Disconnect，即時中止無效推理任務。

詳細測試數據請參閱: [PERFORMANCE.md](./PERFORMANCE.md)

---

## 專案結構

```
lama-triton/
├── gateway/             # 應用網關層
│   ├── app/
│   │   ├── core/        # 影像前/後處理引擎
│   │   ├── schemas/     # 接口資料結構
│   │   └── main.py      # API 進入點
│   └── Dockerfile
├── models/              # 推理引擎模型倉庫 (Model Repository)
│   └── lama_pytorch/
│       ├── config.pbtxt # Node Scheduling 配置
│       └── 1/
├── docker-compose.yml   # 容器編排
├── architecture.md      # 技術架構文件
└── PERFORMANCE.md       # 效能測試報告
```
