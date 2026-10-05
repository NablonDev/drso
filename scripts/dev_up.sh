#!/usr/bin/env bash
# Dev: start Postgres (if down), the API on :8710 and the worker. Logs in data/logs.
set -euo pipefail
cd "$(dirname "$0")/.."
source dev.env
mkdir -p data/logs
pg_ctl -D data/pg/cluster status >/dev/null 2>&1 || pg_ctl -D data/pg/cluster -o "-p 55432 -k /tmp" -l data/pg/pg.log start
pkill -u "$USER" -f "uvicorn beeline_api.app" || true
pkill -u "$USER" -f "beeline_api.worker" || true
sleep 1
nohup uvicorn beeline_api.app:app --host 0.0.0.0 --port 8710 --workers 2 > data/logs/api.log 2>&1 &
nohup python -m beeline_api.worker > data/logs/worker.log 2>&1 &
echo "api :8710, worker started"
