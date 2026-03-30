#!/usr/bin/env bash
# AWS ECS deployment script for Go-gRPC Multi-LLM Platform
# Usage: AWS_REGION=us-east-1 AWS_ACCOUNT_ID=123456789 ./deploy.sh
set -euo pipefail

REGION=${AWS_REGION:-us-east-1}
ACCOUNT_ID=${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID}
ECR_REPO="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/multi-llm-grpc"
MODEL_ID=${MODEL_ID:-google/gemma-3-4b-it}
IMAGE_TAG=${IMAGE_TAG:-latest}

echo "=== Building Docker image ==="
docker build \
    --build-arg MODEL_ID="$MODEL_ID" \
    --build-arg USE_QUANTIZATION=true \
    -t "multi-llm-grpc:$IMAGE_TAG" \
    "$(dirname "$0")/../.."

echo "=== Pushing to ECR ==="
aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$ECR_REPO"

docker tag "multi-llm-grpc:$IMAGE_TAG" "$ECR_REPO:$IMAGE_TAG"
docker push "$ECR_REPO:$IMAGE_TAG"

echo "=== Registering ECS task definition ==="
# Replace placeholders in task definition
sed \
    -e "s|<account-id>|$ACCOUNT_ID|g" \
    -e "s|<region>|$REGION|g" \
    "$(dirname "$0")/ecs-task-definition.json" \
    | aws ecs register-task-definition \
        --cli-input-json file:///dev/stdin

echo "=== Updating ECS service ==="
aws ecs update-service \
    --cluster multi-llm-cluster \
    --service multi-llm-grpc-svc \
    --task-definition multi-llm-grpc \
    --force-new-deployment \
    --region "$REGION"

echo "Deployment initiated. Monitor with:"
echo "  aws ecs describe-services --cluster multi-llm-cluster --services multi-llm-grpc-svc"
