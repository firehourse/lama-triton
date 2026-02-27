# syntax=docker/dockerfile:1
FROM python:3.12-slim

# ---------- uv ----------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# 先 copy 依賴描述檔（利用 layer cache）
COPY pyproject.toml uv.lock ./

# uv sync — 只裝 production 依賴，不裝 dev 工具
RUN uv sync --frozen --no-dev

# Copy 應用程式碼
COPY gateway/ ./gateway/
COPY main.py ./

# uv run 會自動使用 .venv
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080

CMD ["uvicorn", "gateway.app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--timeout-keep-alive", "30"]
# --workers 1: async Gateway 靠 event loop 並發，不需要多 process
# 若將來 CPU 前處理成瓶頸，同步把 Triton count 調高再考慮 workers=2
