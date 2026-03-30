# Multi-stage Dockerfile for Go-gRPC Multi-LLM Research Platform
#
# Supports models:
#   • google/gemma-3-4b-it       (Gemma 3 4B — default)
#   • Qwen/Qwen3-4B              (Qwen3 4B — set MODEL_ID=Qwen/Qwen3-4B)
#   • Qwen/Qwen2.5-3B-Instruct   (Qwen2.5 3B — lighter fallback)
#
# Quantization: NF4 4-bit (bitsandbytes) replicating Google's TurboQuant/E4B.
# Memory: ~2.5GB per 4B model vs ~16GB fp32.
#
# Build args:
#   MODEL_ID            HuggingFace model to pre-download (default: google/gemma-3-4b-it)
#   USE_QUANTIZATION    "true"/"false" (default: true)
#
# ── Stage 1: Python Builder ───────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS python-builder

ARG MODEL_ID=google/gemma-3-4b-it
ARG USE_QUANTIZATION=true

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache
ENV MODEL_ID=${MODEL_ID}
ENV USE_QUANTIZATION=${USE_QUANTIZATION}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY python_model_server/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download model weights into the image layer to reduce cold-start time.
# Skip this step if you prefer to mount a volume with the model cache.
RUN python - <<'EOF'
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id  = os.environ["MODEL_ID"]
cache_dir = os.environ["HF_HOME"]
use_quant = os.environ.get("USE_QUANTIZATION", "true").lower() in ("true","1")

print(f"Pre-downloading: {model_id}  quant={use_quant}")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir)

kwargs = dict(trust_remote_code=True, cache_dir=cache_dir, low_cpu_mem_usage=True)
try:
    import bitsandbytes
    if use_quant:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs.update({"torch_dtype": torch.float32, "device_map": "cpu"})
except ImportError:
    kwargs.update({"torch_dtype": torch.float32, "device_map": "cpu"})

model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
print("Model cached successfully.")
EOF

# ── Stage 2: Go Builder ───────────────────────────────────────────────────────
FROM golang:1.23-bookworm AS go-builder

WORKDIR /app

# Install protoc
RUN apt-get update && apt-get install -y --no-install-recommends unzip curl && \
    curl -sLO https://github.com/protocolbuffers/protobuf/releases/download/v25.3/protoc-25.3-linux-x86_64.zip && \
    unzip -q protoc-25.3-linux-x86_64.zip -d /usr/local && \
    rm protoc-25.3-linux-x86_64.zip

# Install Go proto plugins
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.36.0 && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.6.0

COPY go_grpc_server/go.mod go_grpc_server/go.sum ./
RUN go mod download

# Copy everything (proto + source + sub-packages)
COPY go_grpc_server/ ./

# Regenerate protobuf code
RUN export PATH=$PATH:$(go env GOPATH)/bin && \
    protoc --go_out=. --go_opt=paths=source_relative \
           --go-grpc_out=. --go-grpc_opt=paths=source_relative \
           llm_service.proto

# Build static binary
RUN CGO_ENABLED=0 GOOS=linux go build -a -ldflags="-s -w" -o grpc-server .

# ── Stage 3: Runtime ──────────────────────────────────────────────────────────
FROM ubuntu:24.04

ARG MODEL_ID=google/gemma-3-4b-it

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/cache
ENV MODEL_ID=${MODEL_ID}
ENV USE_QUANTIZATION=true
ENV PYTHON_PORT=8001
ENV PYTHON_HOST=http://localhost:8001
ENV OLLAMA_HOST=http://localhost:11434
ENV PORT=7860
ENV GEMMA_MODEL_ID=${MODEL_ID}
ENV GEMMA_QUANT=int4_nf4

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Optional: install Ollama for local Qwen inference
# Uncomment to bake Ollama into the image (adds ~500MB):
# RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Python dependencies
COPY python_model_server/requirements.txt python_model_server/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r python_model_server/requirements.txt

# Cached model weights from builder
COPY --from=python-builder /app/cache /app/cache

# Python application
COPY python_model_server/app.py python_model_server/

# Go binary
COPY --from=go-builder /app/grpc-server /app/grpc-server

# Orchestration script
COPY run_app.sh .
RUN chmod +x run_app.sh

# Benchmark output directory
RUN mkdir -p /app/benchmarks

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -f http://localhost:8001/health | python3 -c \
        "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d['model_loaded'] else 1)"

CMD ["./run_app.sh"]
