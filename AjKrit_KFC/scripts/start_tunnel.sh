#!/usr/bin/env bash
set -euo pipefail

port="${PORT:-5000}"
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cloudflared_bin="${CLOUDFLARED_BIN:-cloudflared}"

if ! command -v "$cloudflared_bin" >/dev/null 2>&1; then
  if [[ -x "$project_dir/.tools/cloudflared" ]]; then
    cloudflared_bin="$project_dir/.tools/cloudflared"
  else
  echo "cloudflared is not installed. Install it, then run this script again." >&2
  exit 1
  fi
fi

exec "$cloudflared_bin" tunnel --url "http://127.0.0.1:${port}"
