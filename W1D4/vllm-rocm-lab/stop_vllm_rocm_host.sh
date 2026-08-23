#!/usr/bin/env bash
set -euo pipefail

pkill -f ".venv-vllm/bin/vllm serve" || true
