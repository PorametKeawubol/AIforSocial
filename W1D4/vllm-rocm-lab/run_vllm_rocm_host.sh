#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

source W1D4/vllm-rocm-lab/env_vllm_rocm.sh

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
PORT="${PORT:-8000}"

.venv-vllm/bin/vllm serve "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80
