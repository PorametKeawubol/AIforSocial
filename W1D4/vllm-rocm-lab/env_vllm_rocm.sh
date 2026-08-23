#!/usr/bin/env bash

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export VLLM_ROCM_LIB_DIR="$repo_root/W1D4/vllm-rocm-lab/system-libs/root/opt/rocm-7.2.3/lib"
export VLLM_OPENMPI_LIB_DIR="$repo_root/W1D4/vllm-rocm-lab/system-libs/root/usr/lib/x86_64-linux-gnu"
export OLLAMA_ROCM_LIB_DIR="/usr/local/lib/ollama/rocm_v7_2"

export LD_LIBRARY_PATH="$VLLM_ROCM_LIB_DIR:$VLLM_OPENMPI_LIB_DIR:$OLLAMA_ROCM_LIB_DIR:${LD_LIBRARY_PATH:-}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export VLLM_USE_TRITON_FLASH_ATTN="${VLLM_USE_TRITON_FLASH_ATTN:-0}"
