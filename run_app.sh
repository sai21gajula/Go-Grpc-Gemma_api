#!/usr/bin/env bash
# run_app.sh — Orchestration script for the Go-gRPC Multi-LLM Platform
# Starts Python inference server (+ optional Ollama) then launches Go gRPC gateway.
set -euo pipefail

echo "═══════════════════════════════════════════════════════════"
echo "  Go-gRPC Multi-LLM Research Platform"
echo "═══════════════════════════════════════════════════════════"

PYTHON_PORT=${PYTHON_PORT:-8001}
PYTHON_HOST=${PYTHON_HOST:-http://localhost:8001}
OLLAMA_HOST=${OLLAMA_HOST:-http://localhost:11434}
GRPC_PORT=${PORT:-7860}
MAX_STARTUP_WAIT=${MAX_STARTUP_WAIT:-300}
MODEL_ID=${MODEL_ID:-google/gemma-3-4b-it}
ENABLE_OLLAMA=${ENABLE_OLLAMA:-false}
OLLAMA_MODEL=${OLLAMA_MODEL:-qwen3:4b}

echo "  Model:       $MODEL_ID"
echo "  Quant:       ${USE_QUANTIZATION:-true} (NF4/TurboQuant)"
echo "  gRPC port:   $GRPC_PORT"
echo "  Python port: $PYTHON_PORT"
echo "  Ollama:      ${ENABLE_OLLAMA} (model: $OLLAMA_MODEL)"
echo "═══════════════════════════════════════════════════════════"

# ── Cleanup on exit ───────────────────────────────────────────────────────────
PYTHON_PID=""
OLLAMA_PID=""

cleanup() {
    echo "[shutdown] Cleaning up..."
    [[ -n "$PYTHON_PID" ]] && kill "$PYTHON_PID" 2>/dev/null && wait "$PYTHON_PID" 2>/dev/null || true
    [[ -n "$OLLAMA_PID" ]] && kill "$OLLAMA_PID" 2>/dev/null && wait "$OLLAMA_PID" 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# ── Python server ─────────────────────────────────────────────────────────────
echo "[startup] Starting Python inference server (model: $MODEL_ID)..."
cd python_model_server
python app.py > /tmp/python_server.log 2>&1 &
PYTHON_PID=$!
cd ..
echo "[startup] Python PID: $PYTHON_PID"

# Wait for Python server readiness
echo "[startup] Waiting for Python server..."
waited=0
until curl -sf "$PYTHON_HOST/health" | python3 -c \
    "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('status')=='healthy' else 1)" \
    > /dev/null 2>&1; do
    if (( waited >= MAX_STARTUP_WAIT )); then
        echo "[ERROR] Python server not ready after ${MAX_STARTUP_WAIT}s"
        echo "--- Last 30 lines of python_server.log ---"
        tail -30 /tmp/python_server.log 2>/dev/null || true
        exit 1
    fi
    echo "  ...waiting ($waited / $MAX_STARTUP_WAIT s)"
    sleep 5
    waited=$((waited + 5))
done
echo "[startup] Python server is healthy."

# ── Ollama (optional, for Qwen) ───────────────────────────────────────────────
if [[ "$ENABLE_OLLAMA" == "true" ]]; then
    if command -v ollama &>/dev/null; then
        echo "[startup] Starting Ollama..."
        OLLAMA_HOST_ENV=${OLLAMA_HOST#http://} ollama serve > /tmp/ollama.log 2>&1 &
        OLLAMA_PID=$!
        sleep 3
        echo "[startup] Pulling $OLLAMA_MODEL (may take a while on first run)..."
        ollama pull "$OLLAMA_MODEL" || echo "[warn] Failed to pull $OLLAMA_MODEL — Qwen backend may not work"
        echo "[startup] Ollama ready."
    else
        echo "[warn] ENABLE_OLLAMA=true but ollama is not installed — Qwen backend unavailable"
    fi
fi

# ── Go gRPC server ────────────────────────────────────────────────────────────
echo "[startup] Starting Go gRPC server on port $GRPC_PORT..."
export WAIT_FOR_PYTHON=false  # already verified above
exec ./grpc-server
