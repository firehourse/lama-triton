# Deployment Guide: High-Performance Inference System

本文件說明如何將整合好的異步推論系統部署至生產環境。
為求生產環境的絕對穩定性與重現性，建議將所有的依賴映像檔（包含第三方映像檔），皆打包、推送到您專屬的 Private Container Registry，再統一由目標伺服器進行拉取部署。

## 1. 準備生產環境映像檔

請在**建置與推送伺服器（CI/CD Server 或開發用機器）** 上執行以下步驟，以確保映像檔進入您的 Private Registry：

### 1-A. 建置 API Gateway (應用層)
為我們自行開發的 FastAPI Gateway 建立專屬 Image 並推送到 AR：
```bash
# 1. 建立 Image (請在專案根目錄執行)
docker build -t your-private-registry.com/[PROJECT_ID]/[REPO_NAME]/lama-gateway:v1.0.0 -f gateway/Dockerfile .

# 2. 推送至 Registry
docker push your-private-registry.com/[PROJECT_ID]/[REPO_NAME]/lama-gateway:v1.0.0
```

### 1-B. 保留並推送 NVIDIA Inference Engine (運算層)
為了防止未來原廠 Image 版本異動、下架導致服務無法重建，必須將我們測試穩定的依賴版本拉取下來，並同步推送到自有的 Registry：
```bash
# 1. 下載測試穩定的 Triton Image
docker pull nvcr.io/nvidia/tritonserver:24.08-py3

# 2. 改標記為您的 Private Registry Tag
docker tag nvcr.io/nvidia/tritonserver:24.08-py3 your-private-registry.com/[PROJECT_ID]/[REPO_NAME]/tritonserver:24.08-py3

# 3. 推送至 Registry
docker push your-private-registry.com/[PROJECT_ID]/[REPO_NAME]/tritonserver:24.08-py3
```

---

## 2. 準備 Model Weights (權重檔案管理)

生產環境中，「每次容器啟動時動態下載幾 GB 的模型權重」（如開發版 `docker-compose.yml` 的作法）是一件非常高風險的事（網路抖動、下載點失效都會導致服務掛掉）。

最佳實踐是將權重變成不動的靜態擋，有以下兩種策略：

### 策略 A: 預先備妥的主機掛載 (推薦)
在目標的 Production Server 或串接好的 Private Cloud Storage (例如 NFS/S3/GCS 等) 內，**預先建立並下載好資料夾結構**。
```text
/opt/models/                 <-- 這個路徑在目標 Server 必須存在
└── lama_pytorch/
    ├── config.pbtxt
    └── 1/
        └── model.pt         <-- 請先手動/腳本確保這份檔案已存在
```
優點是：Docker 映像檔極小，且抽換權重方便。也是下方 `docker-compose.prod.yml` 預設採用的方式。

### 策略 B: 包裝成特製的 Docker Image (金象檔)
寫一個新的 Dockerfile，以 Triton 為 Base Image，並在 Build 階段把模型 COPY 進去。
優點是：連 Server 都不用去準備資料夾，`docker pull` 下來就自帶模型。缺點是：這個 Image 推送與拉取時會變得非常巨大 (3GB+)。

---

## 3. 目標伺服器 (GPU Instance) 部署範例

在安裝好 `NVIDIA Container Toolkit` 與 `Docker Compose` 的生產環境 Server 上：

1. 準備好上述的 `/opt/models` 目錄與配置。
2. 建立一份**專屬於生產環境的 YAML 檔 (`docker-compose.prod.yml`)**，內容如下：

```yaml
version: '3.8'

services:
  # -----------------------------------------------------------------------
  # C++ Inference Engine (推理引擎)
  # -----------------------------------------------------------------------
  triton-engine:
    image: your-private-registry.com/[PROJECT_ID]/[REPO_NAME]/tritonserver:24.08-py3
    container_name: lama-triton-engine
    shm_size: "2gb"  # 生產環境下加速進程間通訊的關鍵
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - /opt/models:/models  # 直接指向事先備妥的主機路徑或網路硬碟
    command: ["tritonserver", "--model-repository=/models"]
    ports:
      - "8000:8000"
      - "8001:8001"
    restart: always

  # -----------------------------------------------------------------------
  # API Gateway (API 接口)
  # -----------------------------------------------------------------------
  gateway:
    image: your-private-registry.com/[PROJECT_ID]/[REPO_NAME]/lama-gateway:v1.0.0
    container_name: lama-triton-gateway
    environment:
      - TRITON_URL=triton-engine:8001
      - TRITON_MODEL_NAME=lama_pytorch
      # 你可以在此注入其他環境變數，或指向外部的 Logging/APM
    ports:
      - "8090:8080"
    depends_on:
      - triton-engine
    restart: always
```

### 啟動指令

確認伺服器具備存取該 Private Registry 的權限後（例如 `docker login`），即可啟動整座服務：

```bash
docker compose -f docker-compose.prod.yml up -d
```
