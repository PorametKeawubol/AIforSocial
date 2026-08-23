#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
PORT="${PORT:-8000}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-rocm-qwen}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:latest}"

mkdir -p "$HF_CACHE"

if [ ! -e /dev/kfd ] || [ ! -d /dev/dri ]; then
  cat >&2 <<'EOF'
ROCm GPU devices are not visible from this shell.

Expected devices:
  /dev/kfd
  /dev/dri

This usually means Docker Desktop or the current sandbox cannot pass the AMD GPU
to containers. Run this script from a normal Linux host with ROCm + Docker Engine,
not a Docker Desktop VM/context, then try again.
EOF
  exit 1
fi

if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "$CONTAINER_NAME is already running."
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  docker rm "$CONTAINER_NAME" >/dev/null
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size=8g \
  -p "$PORT:8000" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -e HIP_VISIBLE_DEVICES=0 \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  -e VLLM_USE_TRITON_FLASH_ATTN=0 \
  "$IMAGE" \
  --model "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.82

echo "Started $CONTAINER_NAME on http://127.0.0.1:$PORT/v1"
echo "Logs: docker logs -f $CONTAINER_NAME"
