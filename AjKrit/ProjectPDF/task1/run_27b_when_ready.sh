#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OLLAMA_HOST=127.0.0.1:11435
model='hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M'
# Wait for the existing pull; never infer completion from a partial file.
while ! ollama show "$model" >/dev/null 2>&1; do
  sleep 30
done
python3 local_probe.py --model "$model" > artifacts/probe-27b.log 2>&1
