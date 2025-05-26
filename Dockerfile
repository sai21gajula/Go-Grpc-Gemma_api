# Multi-stage Dockerfile for Hugging Face Spaces
# Optimized for CPU Basic tier (2 vCPUs, 16GB RAM)

# Stage 1: Python Builder - Download and cache the model
FROM python:3.10-slim-buster as python-builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache
ENV MODEL_ID=google/gemma-3n-E4B-it-litert-preview

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy Python requirements and install dependencies
COPY python_model_server/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download and cache the model to reduce cold start time
RUN python -c "
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = os.getenv('MODEL_ID', 'google/gemma-3n-E4B-it-litert-preview')
cache_dir = os.getenv('HF_HOME', '/app/cache')

print(f'Pre-downloading model: {model_id}')
print(f'Cache directory: {cache_dir}')

# Download tokenizer
print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    cache_dir=cache_dir
)

# Download model
print('Downloading model...')
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    cache_dir=cache_dir,
    low_cpu_mem_usage=True
)

print('Model and tokenizer cached successfully!')
"

# Stage 2: Go Builder - Build the Go gRPC server
FROM golang:1.20-buster as go-builder

# Install protobuf compiler and plugins
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Install Go protobuf plugins
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@latest && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Set working directory
WORKDIR /app

# Copy Go module files
COPY go_grpc_server/go.mod go_grpc_server/go.sum ./
RUN go mod download

# Copy proto file and generate Go code
COPY go_grpc_server/llm_service.proto .
RUN protoc --go_out=. --go_opt=paths=source_relative \
    --go-grpc_out=. --go-grpc_opt=paths=source_relative \
    llm_service.proto

# Copy Go source code
COPY go_grpc_server/main.go .

# Build the Go binary (statically linked)
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o grpc-server .

# Stage 3: Final Runtime Image
FROM ubuntu:22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache
ENV MODEL_ID=google/gemma-3n-E4B-it-litert-preview
ENV PYTHON_PORT=8001
ENV PYTHON_HOST=http://localhost:8001
ENV PORT=7860
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

# Create app directory
WORKDIR /app

# Copy cached model from python builder
COPY --from=python-builder /app/cache /app/cache

# Copy Python application and install dependencies
COPY python_model_server/requirements.txt python_model_server/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r python_model_server/requirements.txt

COPY python_model_server/app.py python_model_server/

# Copy Go binary from go builder
COPY --from=go-builder /app/grpc-server /app/grpc-server
COPY --from=go-builder /app/pb /app/pb

# Create and copy the orchestration script
COPY run_app.sh .
RUN chmod +x run_app.sh

# Expose the gRPC port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the orchestration script
CMD ["./run_app.sh"]
