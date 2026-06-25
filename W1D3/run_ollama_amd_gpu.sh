#!/usr/bin/env bash
set -euo pipefail

OLLAMA_LOCAL_DIR="${OLLAMA_LOCAL_DIR:-/tmp/ollama-local}"
OLLAMA_BIN="$OLLAMA_LOCAL_DIR/bin/ollama"

if ! command -v zstd >/dev/null 2>&1; then
  echo "zstd is required. Install it first: sudo apt install zstd"
  exit 1
fi

if [ ! -x "$OLLAMA_BIN" ]; then
  mkdir -p "$OLLAMA_LOCAL_DIR"
  curl -L --fail --progress-bar https://ollama.com/download/ollama-linux-amd64.tar.zst \
    | zstd -d \
    | tar -xf - -C "$OLLAMA_LOCAL_DIR"
fi

if command -v docker >/dev/null 2>&1 && docker ps -a --format '{{.Names}}' | grep -qx ollama; then
  docker stop ollama >/dev/null 2>&1 || true
fi

export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
export OLLAMA_LLM_LIBRARY="${OLLAMA_LLM_LIBRARY:-vulkan}"
export GGML_VK_VISIBLE_DEVICES="${GGML_VK_VISIBLE_DEVICES:-0}"

exec "$OLLAMA_BIN" serve
