#!/bin/bash
set -e

echo "=========================================="
echo "Starting Gemma LLM Hybrid Server"
echo "=========================================="

# Environment variables
PYTHON_PORT=${PYTHON_PORT:-8001}
PYTHON_HOST=${PYTHON_HOST:-http://localhost:8001}
GRPC_PORT=${PORT:-7860}
METRICS_PORT=${METRICS_PORT:-9090}
MAX_STARTUP_WAIT=${MAX_STARTUP_WAIT:-300}

echo "Configuration:"
echo "  Python server port: $PYTHON_PORT"
echo "  Python server host: $PYTHON_HOST"
echo "  gRPC server port: $GRPC_PORT"
echo "  Metrics port: $METRICS_PORT"
echo "  Model ID: $MODEL_ID"
echo "  Max startup wait: ${MAX_STARTUP_WAIT}s"
echo "=========================================="

# Function to check if Python server is ready
check_python_server() {
    curl -s -f "$PYTHON_HOST/health" >/dev/null 2>&1
    return $?
}

# Function to wait for Python server with timeout
wait_for_python_server() {
    echo "Waiting for Python server to be ready..."
    local count=0
    local max_attempts=$((MAX_STARTUP_WAIT / 5))
    
    while ! check_python_server; do
        if [ $count -ge $max_attempts ]; then
            echo "ERROR: Python server failed to start within ${MAX_STARTUP_WAIT} seconds"
            echo "Checking Python server logs..."
            if [ -f /tmp/python_server.log ]; then
                tail -20 /tmp/python_server.log
            fi
            exit 1
        fi
        
        count=$((count + 1))
        echo "  Attempt $count/$max_attempts - Python server not ready yet..."
        sleep 5
    done
    
    echo "✅ Python server is ready!"
}

# Function to cleanup background processes
cleanup() {
    echo "Cleaning up..."
    if [ ! -z "$PYTHON_PID" ]; then
        echo "Stopping Python server (PID: $PYTHON_PID)..."
        kill $PYTHON_PID 2>/dev/null || true
        wait $PYTHON_PID 2>/dev/null || true
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Start Python FastAPI server in background
echo "Starting Python model server..."
cd python_model_server
python app.py > /tmp/python_server.log 2>&1 &
PYTHON_PID=$!
cd ..

echo "Python server started with PID: $PYTHON_PID"

# Wait for Python server to be ready
wait_for_python_server

# Verify model is loaded
echo "Verifying model loading status..."
HEALTH_RESPONSE=$(curl -s "$PYTHON_HOST/health" || echo '{"model_loaded": false}')
MODEL_LOADED=$(echo "$HEALTH_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('model_loaded', False))" 2>/dev/null || echo "false")

if [ "$MODEL_LOADED" = "True" ] || [ "$MODEL_LOADED" = "true" ]; then
    echo "✅ Model loaded successfully!"
else
    echo "⚠️  Model may still be loading, but Python server is responsive"
fi

echo "=========================================="
echo "Starting Go gRPC server..."
echo "Server will be available at: 0.0.0.0:$GRPC_PORT"
echo "=========================================="

# Start Go gRPC server in foreground
export WAIT_FOR_PYTHON=false  # Python server is already verified
export METRICS_PORT
exec ./grpc-server
