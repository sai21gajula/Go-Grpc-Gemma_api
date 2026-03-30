#!/usr/bin/env bash
# dev.sh — Development helper for the Go-gRPC Multi-LLM Research Platform
set -euo pipefail

CMD=${1:-help}

# ── Model defaults ──────────────────────────────────────────────────────────
GEMMA_MODEL=${GEMMA_MODEL_ID:-google/gemma-3-4b-it}
QWEN_MODEL=${OLLAMA_MODEL:-qwen3:4b}
GRPC_PORT=${PORT:-7860}
PYTHON_PORT=${PYTHON_PORT:-8001}

case "$CMD" in

# ── proto: regenerate Go protobuf files ──────────────────────────────────────
proto)
    echo "Generating protobuf code..."
    cd go_grpc_server
    export PATH=$PATH:$(go env GOPATH)/bin
    protoc --go_out=. --go_opt=paths=source_relative \
           --go-grpc_out=. --go-grpc_opt=paths=source_relative \
           llm_service.proto
    echo "Protobuf code regenerated."
    ;;

# ── deps: install all dependencies ───────────────────────────────────────────
deps)
    echo "Installing Go dependencies..."
    cd go_grpc_server && go mod tidy && cd ..
    echo "Installing Python dependencies..."
    cd python_model_server && pip install -r requirements.txt && cd ..
    echo "Dependencies installed."
    ;;

# ── python: start Python inference server ────────────────────────────────────
python)
    MODEL_ID=${GEMMA_MODEL} PYTHON_PORT=${PYTHON_PORT} \
    USE_QUANTIZATION=${USE_QUANTIZATION:-true} \
    python python_model_server/app.py
    ;;

# ── python-qwen: start Python server loading Qwen3-4B ────────────────────────
python-qwen)
    MODEL_ID=Qwen/Qwen3-4B PYTHON_PORT=${PYTHON_PORT} \
    USE_QUANTIZATION=${USE_QUANTIZATION:-true} \
    python python_model_server/app.py
    ;;

# ── go: start Go gRPC gateway ─────────────────────────────────────────────────
go)
    cd go_grpc_server
    PORT=${GRPC_PORT} PYTHON_HOST=http://localhost:${PYTHON_PORT} \
    OLLAMA_HOST=http://localhost:11434 \
    GEMMA_MODEL_ID=${GEMMA_MODEL} \
    GEMMA_QUANT=${GEMMA_QUANT:-int4_nf4} \
    GEMINI_API_KEY=${GEMINI_API_KEY:-} \
    go run .
    ;;

# ── ollama-pull: download Qwen3-4B via Ollama ─────────────────────────────────
ollama-pull)
    echo "Pulling ${QWEN_MODEL} via Ollama..."
    ollama pull "${QWEN_MODEL}"
    echo "Model ready."
    ;;

# ── build: build Docker image ─────────────────────────────────────────────────
build)
    MODEL=${BUILD_MODEL:-google/gemma-3-4b-it}
    echo "Building Docker image with model: $MODEL"
    docker build \
        --build-arg MODEL_ID="$MODEL" \
        --build-arg USE_QUANTIZATION=true \
        -t multi-llm-grpc:latest .
    echo "Image built: multi-llm-grpc:latest"
    ;;

# ── run: run Docker container ─────────────────────────────────────────────────
run)
    docker run --rm \
        -p 7860:7860 \
        -e GEMINI_API_KEY="${GEMINI_API_KEY:-}" \
        -e ENABLE_OLLAMA=false \
        multi-llm-grpc:latest
    ;;

# ── health: check all services ───────────────────────────────────────────────
health)
    echo "=== Python server health ==="
    curl -sf http://localhost:${PYTHON_PORT}/health | python3 -m json.tool || echo "Python server not reachable"
    echo ""
    echo "=== gRPC services ==="
    grpcurl -plaintext localhost:${GRPC_PORT} list 2>/dev/null || echo "gRPC server not reachable (is it running?)"
    echo ""
    echo "=== Ollama ==="
    curl -sf http://localhost:11434/api/tags 2>/dev/null | python3 -m json.tool | head -20 || echo "Ollama not running"
    ;;

# ── test-gemma: unary call to Gemma ──────────────────────────────────────────
test-gemma)
    grpcurl -plaintext -d '{
      "prompt": "Explain what TurboQuant NF4 quantization is in 2 sentences.",
      "model_id": "gemma:'"${GEMMA_MODEL}"'",
      "temperature": 0.7,
      "max_new_tokens": 80
    }' "localhost:${GRPC_PORT}" llm_service.LLMService/GenerateText
    ;;

# ── test-qwen: unary call to Qwen via Ollama ─────────────────────────────────
test-qwen)
    grpcurl -plaintext -d '{
      "prompt": "Why is Go better than Python for high-concurrency gRPC servers?",
      "model_id": "qwen:'"${QWEN_MODEL}"'",
      "temperature": 0.7,
      "max_new_tokens": 80
    }' "localhost:${GRPC_PORT}" llm_service.LLMService/GenerateText
    ;;

# ── test-stream: streaming call to Gemma ─────────────────────────────────────
test-stream)
    grpcurl -plaintext -d '{
      "prompt": "Tell me a short story about a robot that learned to dream.",
      "model_id": "gemma:'"${GEMMA_MODEL}"'",
      "temperature": 0.8,
      "max_new_tokens": 150
    }' "localhost:${GRPC_PORT}" llm_service.LLMService/StreamGenerateText
    ;;

# ── test-evaluate: EvaluateGeneration RPC ────────────────────────────────────
test-evaluate)
    grpcurl -plaintext -d '{
      "prompt": "What is 2 + 2?",
      "reference_answer": "4",
      "model_ids": ["gemma:'"${GEMMA_MODEL}"'"],
      "temperature": 0.0,
      "max_new_tokens": 10
    }' "localhost:${GRPC_PORT}" llm_service.LLMService/EvaluateGeneration
    ;;

# ── benchmark: run BenchmarkModels RPC ───────────────────────────────────────
benchmark)
    echo "Running benchmark across models..."
    grpcurl -plaintext -d '{
      "prompts": [
        "What is machine learning?",
        "Explain the difference between gRPC and REST.",
        "What is 4-bit NF4 quantization?"
      ],
      "model_ids": ["gemma:'"${GEMMA_MODEL}"'"],
      "runs_per_prompt": 2,
      "include_rest_comparison": true,
      "temperature": 0.0,
      "max_new_tokens": 80
    }' "localhost:${GRPC_PORT}" llm_service.LLMService/BenchmarkModels
    ;;

# ── rest-test: direct REST call to Python (bypass gRPC) ──────────────────────
rest-test)
    curl -s -X POST http://localhost:${PYTHON_PORT}/predict \
        -H "Content-Type: application/json" \
        -d '{
          "prompt": "What is gRPC?",
          "model_id": "'"${GEMMA_MODEL}"'",
          "temperature": 0.0,
          "max_new_tokens": 60
        }' | python3 -m json.tool
    ;;

# ── eval-test: test /evaluate endpoint directly ──────────────────────────────
eval-test)
    curl -s -X POST http://localhost:${PYTHON_PORT}/evaluate \
        -H "Content-Type: application/json" \
        -d '{
          "generated": "The quick brown fox jumps over the lazy dog.",
          "reference": "A fast fox jumped over a sleeping dog."
        }' | python3 -m json.tool
    ;;

# ── help ──────────────────────────────────────────────────────────────────────
help|*)
    cat <<'HELP'
Go-gRPC Multi-LLM Research Platform — Dev Helper

USAGE:  ./dev.sh <command> [env overrides]

SETUP:
  deps          Install Go + Python dependencies
  proto         Regenerate protobuf Go code
  ollama-pull   Download Qwen3-4B via Ollama

SERVERS:
  python        Start Python server (Gemma 3-4B, NF4 quant)
  python-qwen   Start Python server (Qwen3-4B, NF4 quant)
  go            Start Go gRPC gateway

TESTING:
  health        Check all services
  test-gemma    Unary call → Gemma (via Go gRPC → Python)
  test-qwen     Unary call → Qwen  (via Go gRPC → Ollama)
  test-stream   Streaming call → Gemma
  test-evaluate EvaluateGeneration RPC (BLEU + ROUGE + latency)
  benchmark     BenchmarkModels RPC (multi-model, gRPC vs REST comparison)
  rest-test     Direct REST call to Python (bypasses gRPC — for comparison)
  eval-test     Direct BLEU/ROUGE test on Python /evaluate endpoint

DOCKER:
  build         Build Docker image (BUILD_MODEL=... to choose model)
  run           Run Docker container on port 7860

TYPICAL WORKFLOW:
  1.  ./dev.sh deps
  2a. ./dev.sh python          # Terminal 1: Gemma
   OR ./dev.sh ollama-pull && ollama serve &  # for Qwen
  3.  ./dev.sh go              # Terminal 2: gRPC gateway
  4.  ./dev.sh health          # Terminal 3: verify
  5.  ./dev.sh benchmark       # Run full benchmark
  6.  cat benchmarks/results_*.json | python3 -m json.tool

ENV VARS:
  GEMMA_MODEL_ID    HuggingFace model ID (default: google/gemma-3-4b-it)
  GEMINI_API_KEY    Google Gemini API key (optional; enables Gemini backend)
  USE_QUANTIZATION  true/false (default: true — NF4 4-bit)
  PORT              gRPC port (default: 7860)
  PYTHON_PORT       Python server port (default: 8001)
HELP
    ;;
esac
