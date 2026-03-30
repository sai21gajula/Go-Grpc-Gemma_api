#!/usr/bin/env bash
# Azure AKS deployment script for Go-gRPC Multi-LLM Platform
# Usage: RESOURCE_GROUP=myRG ACR_NAME=myACR AKS_CLUSTER=myAKS ./deploy.sh
set -euo pipefail

RESOURCE_GROUP=${RESOURCE_GROUP:?Set RESOURCE_GROUP}
ACR_NAME=${ACR_NAME:?Set ACR_NAME}
AKS_CLUSTER=${AKS_CLUSTER:?Set AKS_CLUSTER}
MODEL_ID=${MODEL_ID:-google/gemma-3-4b-it}
IMAGE_TAG=${IMAGE_TAG:-latest}
IMAGE="$ACR_NAME.azurecr.io/multi-llm-grpc:$IMAGE_TAG"

echo "=== Building Docker image ==="
docker build \
    --build-arg MODEL_ID="$MODEL_ID" \
    --build-arg USE_QUANTIZATION=true \
    -t "multi-llm-grpc:$IMAGE_TAG" \
    "$(dirname "$0")/../.."

echo "=== Pushing to Azure Container Registry ==="
az acr login --name "$ACR_NAME"
docker tag "multi-llm-grpc:$IMAGE_TAG" "$IMAGE"
docker push "$IMAGE"

echo "=== Getting AKS credentials ==="
az aks get-credentials \
    --resource-group "$RESOURCE_GROUP" \
    --name "$AKS_CLUSTER" \
    --overwrite-existing

echo "=== Creating secrets ==="
kubectl create secret generic llm-secrets \
    --from-literal=GEMINI_API_KEY="${GEMINI_API_KEY:-}" \
    --dry-run=client -o yaml | kubectl apply -f -

echo "=== Updating image in deployment ==="
# Replace image placeholder in deployment YAML and apply
sed "s|<your-acr>.azurecr.io/multi-llm-grpc:latest|$IMAGE|g" \
    "$(dirname "$0")/aks-deployment.yaml" \
    | kubectl apply -f -

echo "=== Waiting for rollout ==="
kubectl rollout status deployment/multi-llm-grpc --timeout=600s

echo "=== Getting external IP ==="
kubectl get service multi-llm-grpc-svc

echo ""
echo "Once EXTERNAL-IP is assigned, test with:"
echo "  grpcurl -plaintext <EXTERNAL-IP>:7860 llm_service.LLMService/GenerateText"
