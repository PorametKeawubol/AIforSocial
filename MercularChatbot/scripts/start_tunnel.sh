#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
port="${PORT:-}"

# Match Settings.from_env() without sourcing arbitrary shell from .env.
if [[ -z "$port" && -f "$project_dir/.env" ]]; then
  while IFS='=' read -r key value; do
    key="${key//[[:space:]]/}"
    if [[ "$key" == "PORT" ]]; then
      value="${value%%#*}"
      value="${value//[[:space:]\"\']/}"
      port="$value"
      break
    fi
  done < "$project_dir/.env"
fi

port="${port:-5000}"
if ! [[ "$port" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
  echo "PORT must be an integer from 1 to 65535" >&2
  exit 1
fi

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared was not found. Install it from https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/" >&2
  exit 1
fi

exec cloudflared tunnel --url "http://127.0.0.1:${port}"
