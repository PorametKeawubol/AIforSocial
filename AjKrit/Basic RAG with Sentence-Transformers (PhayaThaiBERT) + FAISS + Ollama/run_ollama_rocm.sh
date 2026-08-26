#!/usr/bin/env bash
set -euo pipefail

# RX 7600S is GPU 0 on this machine.  Do not load the integrated Radeon 680M.
export HIP_VISIBLE_DEVICES=0
exec ollama serve
