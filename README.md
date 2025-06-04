---
title: Gemma 3B LLM gRPC API
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
cpu: 2
memory: 16
gpu: false
models:
  - google/gemma-3n-E4B-it-litert-preview
---

# Go-gRPC-Gemma API

High-performance LLM serving with hybrid Go/Python architecture for Hugging Face Spaces.

## 🚀 Features

- **Hybrid Architecture**: Go gRPC server + Python FastAPI model server
- **High Concurrency**: Efficient handling of multiple simultaneous requests
- **Streaming Support**: Real-time token-by-token text generation
- **Resource Optimized**: Designed for CPU Basic tier (2 vCPUs, 16GB RAM)
- **Production Ready**: Comprehensive error handling and logging
- **Prometheus Metrics**: Exposes request and token statistics on `/metrics`
- **Flask Frontend**: Simple web UI for manual testing

## 📋 API Reference

### gRPC Service: `LLMService`

#### Methods

1. **GenerateText** (Unary RPC)
   - Request: `GenerateRequest`
   - Response: `GenerateResponse`

2. **StreamGenerateText** (Server Streaming RPC)
   - Request: `GenerateRequest`
   - Response: Stream of `StreamGenerateResponse`

#### Message Types

```protobuf
message GenerateRequest {
  string prompt = 1;
  string model_id = 2;
  float temperature = 3;
  int32 max_new_tokens = 4;
}

message GenerateResponse {
  string generated_text = 1;
}

message StreamGenerateResponse {
  string partial_text = 1;
  bool done = 2;
}
```

## 🛠️ Usage Examples

### Using grpcurl

#### Unary Generation
```bash
grpcurl -plaintext -d '{
  "prompt": "What is artificial intelligence?",
  "model_id": "google/gemma-3n-E4B-it-litert-preview",
  "temperature": 0.7,
  "max_new_tokens": 100
}' your-space-name.hf.space:443 llm_service.LLMService/GenerateText
```

#### Streaming Generation
```bash
grpcurl -plaintext -d '{
  "prompt": "Explain quantum computing in simple terms:",
  "model_id": "google/gemma-3n-E4B-it-litert-preview",
  "temperature": 0.8,
  "max_new_tokens": 200
}' your-space-name.hf.space:443 llm_service.LLMService/StreamGenerateText
```

### For Local Development
```bash
# Unary call
grpcurl -plaintext -d '{
  "prompt": "Hello, how are you?",
  "model_id": "google/gemma-3n-E4B-it-litert-preview",
  "temperature": 0.7,
  "max_new_tokens": 50
}' localhost:7860 llm_service.LLMService/GenerateText

# Streaming call
grpcurl -plaintext -d '{
  "prompt": "Tell me a story about space exploration:",
  "model_id": "google/gemma-3n-E4B-it-litert-preview",
  "temperature": 0.9,
  "max_new_tokens": 150
}' localhost:7860 llm_service.LLMService/StreamGenerateText
```

## 🏗️ Architecture

### Components

1. **Go gRPC Server** (`go_grpc_server/`)
   - High-performance gRPC API gateway
   - Handles concurrent client connections
   - Proxies requests to Python model server

2. **Python FastAPI Model Server** (`python_model_server/`)
   - Loads and serves the Gemma model
   - Provides HTTP/JSON API internally
   - Handles model inference and streaming

3. **Docker Multi-Stage Build**
   - Pre-downloads model for faster cold starts
   - Optimized for Hugging Face Spaces
   - Efficient resource utilization

### Data Flow

```
Flutter App → gRPC → Go Server → HTTP → Python Server → Model → Response
```

## 🚀 Local Development

### Prerequisites
- Go 1.20+
- Python 3.10+
- protobuf compiler
- Docker (optional)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Go-Grpc-Gemma_api
```

2. **Generate protobuf code**
```bash
cd go_grpc_server
protoc --go_out=. --go_opt=paths=source_relative \
       --go-grpc_out=. --go-grpc_opt=paths=source_relative \
       llm_service.proto
```

3. **Install Python dependencies**
```bash
cd python_model_server
pip install -r requirements.txt
```

4. **Run Python server**
```bash
cd python_model_server
python app.py
```

5. **Run Go server** (in another terminal)
```bash
cd go_grpc_server
go run main.go
```

### Using Docker

```bash
# Build the image
docker build -t gemma-grpc-api .

# Run the container
docker run -p 7860:7860 gemma-grpc-api
```

## 📊 Performance Characteristics

- **Cold Start**: ~60-120 seconds (model loading)
- **Inference Latency**: ~500ms-2s per request (CPU dependent)
- **Concurrent Requests**: Handles 10+ simultaneous connections efficiently
- **Memory Usage**: ~8-12GB (model + overhead)
- **CPU Usage**: Scales with request load

## 🔧 Configuration

### Environment Variables

- `PORT`: gRPC server port (default: 7860)
- `PYTHON_PORT`: Python server port (default: 8001)
- `PYTHON_HOST`: Python server URL (default: http://localhost:8001)
- `MODEL_ID`: Model identifier (default: google/gemma-3n-E4B-it-litert-preview)
- `HF_HOME`: Hugging Face cache directory
- `MAX_STARTUP_WAIT`: Maximum startup wait time in seconds (default: 300)
- `METRICS_PORT`: Prometheus metrics port for the Go server (default: 9090)
- `GRPC_HOST`: gRPC endpoint for the Flask frontend (default: localhost:7860)

Metrics are available at `http://localhost:$METRICS_PORT/metrics` when running locally.

## 🐛 Troubleshooting

### Common Issues

1. **Model loading timeout**
   - Increase `MAX_STARTUP_WAIT`
   - Check available memory
   - Verify model ID is correct

2. **Connection refused**
   - Ensure Python server is running
   - Check `PYTHON_HOST` configuration
   - Verify firewall settings

3. **Out of memory**
   - Use smaller `max_new_tokens`
   - Reduce concurrent requests
   - Check available RAM

### Health Checks

```bash
# Check Python server health
curl http://localhost:8001/health

# Test gRPC server
grpcurl -plaintext localhost:7860 list
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 References

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [gRPC Go Tutorial](https://grpc.io/docs/languages/go/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Gemma Model](https://huggingface.co/google/gemma-3n-E4B-it-litert-preview)