# Basic Lab: OpenClaw + Ollama gemma3:12b + Telegram (Docker only)

ชุดนี้ปรับจากไฟล์ lab ให้เหมาะกับ Ubuntu/WSL และใช้ Docker เท่านั้นสำหรับ OpenClaw Gateway/CLI

## สิ่งที่ตั้งค่าไว้

- OpenClaw รันใน Docker Compose
- Ollama ใช้โมเดล local `gemma3:12b`
- OpenClaw container เรียก Ollama host ผ่าน `http://host.docker.internal:11434`
- Telegram เปิดผ่าน `TELEGRAM_BOT_TOKEN`
- Agent หลักชื่อ `PSU Research Assistant`

## วิธีรัน

```bash
cd W1D4/openclaw-docker
cp .env.example .env
openssl rand -hex 32
```

นำค่าที่ generate ไปใส่ `OPENCLAW_GATEWAY_TOKEN` ใน `.env` แล้วใส่ token จาก `@BotFather` ที่ `TELEGRAM_BOT_TOKEN`

```bash
./run.sh
```

ถ้าต้องการให้ `gemma3:12b` ใช้ AMD GPU ให้รัน Ollama บน host ด้วย ROCm bundle ก่อน:

```bash
./start_ollama_rocm.sh
```

ตรวจว่าใช้ GPU:

```bash
./ollama-local/root/bin/ollama ps
rocm-smi --showmeminfo vram --showuse
```

บนเครื่องนี้ Docker Desktop ถูกลด memory เหลือ 8192 MiB เพื่อคืน RAM ให้ Ollama เพราะ `gemma3:12b` ต้องใช้ system memory ร่วมกับ VRAM.

เปิด Control UI:

```text
http://127.0.0.1:18789
```

ใช้ค่า `OPENCLAW_GATEWAY_TOKEN` จาก `.env` เป็น shared secret

## Pair Telegram DM

หลัง gateway start แล้ว ให้ DM bot ใน Telegram ก่อน จากนั้นรัน:

```bash
docker compose run --rm openclaw-cli pairing list telegram
docker compose run --rm openclaw-cli pairing approve telegram <CODE>
```

## ทดสอบ Ollama model

```bash
docker compose run --rm openclaw-cli models list --provider ollama
```

## Prompt สำหรับส่งใน Telegram

ใช้ prompt จาก `prompts/psu-research-assistant.md`

วาง PDF งานวิจัยไว้ที่:

```text
W1D4/openclaw-docker/workspace/Research
```

ผลลัพธ์ที่ต้องส่งใน assignment:

- Screenshot การทำงานใน Telegram หรือ Control UI
- Configuration file: `config/openclaw.json`
- Prompt: `prompts/psu-research-assistant.md`
- Report: `workspace/summary.md`
