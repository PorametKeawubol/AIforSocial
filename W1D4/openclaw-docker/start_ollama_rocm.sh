#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

root_dir="$PWD/ollama-local/root"
ollama_bin="$root_dir/bin/ollama"

if [ ! -x "$ollama_bin" ]; then
  mkdir -p ollama-local "$root_dir"
  curl -L --fail https://ollama.com/download/ollama-linux-amd64.tar.zst \
    -o ollama-local/ollama-linux-amd64.tar.zst
  curl -L --fail https://ollama.com/download/ollama-linux-amd64-rocm.tar.zst \
    -o ollama-local/ollama-linux-amd64-rocm.tar.zst
  tar --zstd -xf ollama-local/ollama-linux-amd64.tar.zst -C "$root_dir"
  tar --zstd -xf ollama-local/ollama-linux-amd64-rocm.tar.zst -C "$root_dir"
fi

systemctl --user stop ollama-rocm.service 2>/dev/null || true

systemd-run --user \
  --unit=ollama-rocm \
  --same-dir \
  --setenv=OLLAMA_HOST=0.0.0.0:11434 \
  --setenv=OLLAMA_DEBUG=1 \
  --setenv=HIP_VISIBLE_DEVICES=0 \
  --setenv=HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  --setenv=OLLAMA_MODELS="$HOME/.ollama/models" \
  "$ollama_bin" serve

for _ in {1..30}; do
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

curl -fsS http://127.0.0.1:11434/api/tags | jq -r '.models[].name'
systemctl --user status ollama-rocm --no-pager | sed -n '1,40p'

