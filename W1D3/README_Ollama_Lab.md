# Ollama LLM Localhost Python Lab

## 1. ตรวจสอบ Docker

```bash
docker --version
docker ps
```

## 2. เตรียม Ollama Container

```bash
docker pull ollama/ollama
docker run -d --name ollama -p 11434:11434 -v ollama:/root/.ollama ollama/ollama
```

ถ้ามี container ชื่อ `ollama` อยู่แล้ว:

```bash
docker start ollama
```

## 3. ดาวน์โหลดโมเดล

```bash
docker exec -it ollama ollama pull qwen3:8b
docker exec -it ollama ollama pull gemma3:12b
docker exec -it ollama ollama pull llama3.1:8b
docker exec -it ollama ollama list
```

## 4. ติดตั้ง Python Library

```bash
python3 -m pip install requests
```

## 5. ทดสอบ REST API

```bash
python3 W1D3/ollama_api_test.py
```

## 6. เปรียบเทียบโมเดล

```bash
python3 W1D3/ollama_model_compare.py
```

ผลลัพธ์จะถูกบันทึกที่:

```text
W1D3/ollama_model_compare_results.md
```
