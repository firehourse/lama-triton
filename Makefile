.PHONY: setup run stop clean test benchmark

# 預設行為：啟動所有服務
run:
	docker-compose up --build -d

# 停止並移除容器
stop:
	docker-compose down

# 重新建立與啟動
restart: stop run

# 檢查日誌
logs:
	docker-compose logs -f

# 執行集成測試 (需先啟動服務)
test:
	uv run python -c "import requests; r = requests.get('http://localhost:8090/health'); print(r.json())"

# 清理下載的模型與快取
clean:
	rm -rf __pycache__ .pytest_cache
	@echo "Note: models/ content is preserved. Delete manually if needed."

# 效能測試 (注意: scripts/ 在 gitignore 中，本地開發可用)
benchmark:
	if [ -f scripts/benchmark.py ]; then \
		uv run python scripts/benchmark.py --workers 6 --target all; \
	else \
		echo "Benchmark script not found (it might be gitignored)."; \
	fi
