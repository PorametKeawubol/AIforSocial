#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose v2 is required." >&2
  exit 1
fi

if ! curl -fsS http://127.0.0.1:11434/api/tags | grep -q '"gemma3:12b"'; then
  echo "Ollama is not reachable at 127.0.0.1:11434 or gemma3:12b is missing." >&2
  echo "Run: ollama pull gemma3:12b" >&2
  exit 1
fi

if [ ! -f .env ]; then
  cp .env.example .env
  token="$(openssl rand -hex 32 2>/dev/null || date +%s%N)"
  sed -i "s/^OPENCLAW_GATEWAY_TOKEN=.*/OPENCLAW_GATEWAY_TOKEN=${token}/" .env
  echo "Created .env. Add TELEGRAM_BOT_TOKEN before rerunning." >&2
  exit 1
fi

if grep -q '^TELEGRAM_BOT_TOKEN=replace-with-botfather-token$' .env || grep -q '^TELEGRAM_BOT_TOKEN=$' .env; then
  echo "Set TELEGRAM_BOT_TOKEN in .env before starting Telegram integration." >&2
  exit 1
fi

mkdir -p state workspace/Research
cp config/openclaw.json state/openclaw.json

if ! docker image inspect openclaw:local >/dev/null 2>&1; then
  if [ ! -d openclaw-src/.git ]; then
    git clone --depth 1 https://github.com/openclaw/openclaw.git openclaw-src
  fi
  docker build -t openclaw:local openclaw-src
fi

docker compose up -d openclaw-gateway
docker compose ps

echo
echo "OpenClaw Control UI: http://127.0.0.1:18789"
echo "Use OPENCLAW_GATEWAY_TOKEN from .env as the shared secret."

