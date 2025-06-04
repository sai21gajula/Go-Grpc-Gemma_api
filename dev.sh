#!/bin/bash

# Development helper script for local testing

set -e

COMMAND=${1:-help}

case $COMMAND in
    "proto")
        echo "Generating protobuf code..."
        cd go_grpc_server
        protoc --go_out=. --go_opt=paths=source_relative \
               --go-grpc_out=. --go-grpc_opt=paths=source_relative \
               llm_service.proto
        echo "✅ Protobuf code generated"
        ;;
    
    "python")
        echo "Starting Python model server..."
        cd python_model_server
        export PYTHON_PORT=8001
        export MODEL_ID=google/gemma-3n-E4B-it-litert-preview
        python app.py
        ;;
    
    "go")
        echo "Starting Go gRPC server..."
        cd go_grpc_server
        export PORT=7860
        export PYTHON_HOST=http://localhost:8001
        export METRICS_PORT=9090
        export WAIT_FOR_PYTHON=false
        go run .
        ;;
    
    "deps")
        echo "Installing dependencies..."
        
        # Install Go dependencies
        echo "Installing Go dependencies..."
        cd go_grpc_server
        go mod tidy
        cd ..
        
        # Install Python dependencies
        echo "Installing Python dependencies..."
        cd python_model_server
        pip install -r requirements.txt
        cd ..
        
        echo "✅ Dependencies installed"
        ;;
    
    "build")
        echo "Building Docker image..."
        docker build -t gemma-grpc-api .
        echo "✅ Docker image built"
        ;;
    
    "run")
        echo "Running Docker container..."
        docker run -p 7860:7860 gemma-grpc-api
        ;;
    
    "test-unary")
        echo "Testing unary gRPC call..."
        grpcurl -plaintext -d '{
          "prompt": "What is artificial intelligence?",
          "model_id": "google/gemma-3n-E4B-it-litert-preview",
          "temperature": 0.7,
          "max_new_tokens": 50
        }' localhost:7860 llm_service.LLMService/GenerateText
        ;;
    
    "test-stream")
        echo "Testing streaming gRPC call..."
        grpcurl -plaintext -d '{
          "prompt": "Tell me a short story:",
          "model_id": "google/gemma-3n-E4B-it-litert-preview",
          "temperature": 0.8,
          "max_new_tokens": 100
        }' localhost:7860 llm_service.LLMService/StreamGenerateText
        ;;
    
    "health")
        echo "Checking service health..."
        echo "Python server health:"
        curl -s http://localhost:8001/health | python3 -m json.tool
        echo -e "\ngRPC server services:"
        grpcurl -plaintext localhost:7860 list
        ;;
    
    "clean")
        echo "Cleaning up..."
        docker system prune -f
        echo "✅ Cleanup complete"
        ;;
    
    "help"|*)
        echo "🚀 Gemma gRPC API Development Helper"
        echo ""
        echo "Usage: ./dev.sh <command>"
        echo ""
        echo "Commands:"
        echo "  proto       - Generate protobuf Go code"
        echo "  deps        - Install all dependencies"
        echo "  python      - Start Python model server"
        echo "  go          - Start Go gRPC server"
        echo "  build       - Build Docker image"
        echo "  run         - Run Docker container"
        echo "  test-unary  - Test unary gRPC call"
        echo "  test-stream - Test streaming gRPC call"
        echo "  health      - Check service health"
        echo "  clean       - Clean Docker artifacts"
        echo "  help        - Show this help"
        echo ""
        echo "Development workflow:"
        echo "  1. ./dev.sh deps     # Install dependencies"
        echo "  2. ./dev.sh proto    # Generate protobuf code"
        echo "  3. ./dev.sh python   # Start Python server (terminal 1)"
        echo "  4. ./dev.sh go       # Start Go server (terminal 2)"
        echo "  5. ./dev.sh test-unary  # Test the API"
        ;;
esac
