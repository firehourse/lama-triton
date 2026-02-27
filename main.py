"""
lama-triton 入口點
用法: uv run uvicorn main:app --host 0.0.0.0 --port 8090 --reload
"""
from gateway.app.main import app  # noqa: F401 — re-export for uvicorn

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8090,
        reload=True,
        loop="uvloop",
    )
