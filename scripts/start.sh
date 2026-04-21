#!/bin/bash
echo "Starting SentinelMD..."
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri ./logs/mlflow &
uvicorn src.api.main:app --reload --port 8000 &
cd frontend && npm start