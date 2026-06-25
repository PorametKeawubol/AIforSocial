# 🧪 Sheet Lab: การใช้งาน Ollama LLM บน Docker ด้วย Python REST API

## 🎯 วัตถุประสงค์

* เรียนรู้การใช้งาน Docker Container
* ติดตั้งและใช้งาน Ollama ผ่าน Docker
* ดาวน์โหลดและใช้งาน Large Language Models (LLMs)
* เรียกใช้งานโมเดลผ่าน Python REST API
* เปรียบเทียบประสิทธิภาพของโมเดล LLM หลายประเภท

---

# ✅ ขั้นตอนที่ 1: ติดตั้ง Docker

ตรวจสอบว่า Docker พร้อมใช้งาน

```bash
docker --version
```

ตรวจสอบ Docker Service

```bash
docker ps
```

หากยังไม่ได้ติดตั้ง

https://docs.docker.com/get-docker/

---

# ✅ ขั้นตอนที่ 2: ดาวน์โหลด Ollama Docker Image

```bash
docker pull ollama/ollama
```

ตรวจสอบ Image

```bash
docker images
```

---

# ✅ ขั้นตอนที่ 3: สร้าง Ollama Container

## Linux (AMD GPU)

```bash
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama:/root/.ollama \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  ollama/ollama
```

## Windows / CPU Mode

```bash
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama:/root/.ollama \
  ollama/ollama
```

ตรวจสอบ Container

```bash
docker ps
```

ผลลัพธ์ควรปรากฏ Container ชื่อ

```text
ollama
```

---

# ✅ ขั้นตอนที่ 4: ดาวน์โหลดโมเดลสำหรับการทดลอง

เข้าสู่ Container

```bash
docker exec -it ollama bash
```

ดาวน์โหลดโมเดล

## Qwen3 8B

```bash
ollama pull qwen3:8b
```

## Gemma3 12B

```bash
ollama pull gemma3:12b
```

## Llama3.1 8B

```bash
ollama pull llama3.1:8b
```

ตรวจสอบรายการโมเดล

```bash
ollama list
```

---

# ✅ ขั้นตอนที่ 5: ทดสอบโมเดลผ่าน Terminal

## Qwen3

```bash
docker exec -it ollama ollama run qwen3:8b
```

## Gemma3

```bash
docker exec -it ollama ollama run gemma3:12b
```

## Llama3.1

```bash
docker exec -it ollama ollama run llama3.1:8b
```

---

# ✅ ขั้นตอนที่ 6: เรียกใช้งาน Ollama ผ่าน REST API

ติดตั้ง Python Library

```bash
pip install requests
```

ตัวอย่างโปรแกรม

```python
import requests

url = "http://localhost:11434/api/generate"

response = requests.post(
    url,
    json={
        "model": "qwen3:8b",
        "prompt": "ประเทศไทยมีกี่จังหวัด",
        "stream": False
    }
)

print(response.json()["response"])
```

---

# 🧠 กิจกรรมที่ 1: เปรียบเทียบประสิทธิภาพโมเดล

ให้ทดลองใช้โมเดล

* qwen3:8b
* gemma3:12b
* llama3.1:8b

โดยใช้คำถามเดียวกัน

## ชุดคำถาม

1. ประเทศไทยมีกี่จังหวัด
2. อธิบายการทำงานของ VLAN และ OSPF
3. เขียน Python Bubble Sort
4. สรุปเนื้อหา Animal Farm
5. วิเคราะห์ผลลัพธ์ Nmap Scan

---

## ตารางบันทึกผล

| หัวข้อ           | Qwen3:8B | Gemma3:12B | Llama3.1:8B |
| ---------------------- | -------- | ---------- | ----------- |
| ความถูกต้อง |          |            |             |
| ภาษาไทย         |          |            |             |
| Coding                 |          |            |             |
| ความเร็ว       |          |            |             |
| ความครบถ้วน |          |            |             |

---

# 🧠 กิจกรรมที่ 2: ตรวจสอบการใช้งาน GPU

## Linux AMD

ติดตั้ง

```bash
sudo apt install radeontop -y
```

ตรวจสอบการใช้งาน GPU

```bash
radeontop
```

หรือ

```bash
watch -n 1 radeontop
```

---

## NVIDIA

```bash
watch -n 1 nvidia-smi
```

สังเกตค่า

* GPU Utilization
* GPU Memory Usage

ขณะรันโมเดล

---

# 📝 สรุปผลการทดลอง

ให้นักศึกษาสรุป

1. โมเดลใดตอบคำถามได้ดีที่สุด
2. โมเดลใดตอบเร็วที่สุด
3. โมเดลใดเหมาะกับเครื่องของตนเองมากที่สุด
4. ปัญหาที่พบระหว่างการทดลอง
5. ข้อดีและข้อเสียของการใช้งาน LLM ผ่าน Docker

---

# ⭐ คำถามท้าย Lab

1. Docker มีข้อดีอย่างไรเมื่อเทียบกับการติดตั้ง Ollama แบบ Native
2. Ollama ใช้พอร์ตใดสำหรับ REST API
3. Container และ Image แตกต่างกันอย่างไร
4. Qwen3, Gemma3 และ Llama3.1 แตกต่างกันอย่างไร
5. หากต้องการสร้าง Chatbot ด้วย Python และ Ollama ต้องมีองค์ประกอบใดบ้าง

## 🔧 หมายเหตุสำหรับ Linux (AMD GPU)

Lab นี้ดำเนินการบนระบบปฏิบัติการ Ubuntu Linux 24.04 และใช้งาน AMD Radeon RX 7600S

### ตรวจสอบ Docker

```bash
docker --version
docker ps
```

### ดาวน์โหลด Ollama Docker Image

```bash
docker pull ollama/ollama
```

### สร้าง Ollama Container พร้อมรองรับ AMD GPU

```bash
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama:/root/.ollama \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --restart unless-stopped \
  ollama/ollama
```

### ตรวจสอบสถานะ Container

```bash
docker ps
```

### ดาวน์โหลดโมเดล

```bash
docker exec -it ollama ollama pull qwen3:8b

docker exec -it ollama ollama pull gemma3:12b

docker exec -it ollama ollama pull llama3.1:8b
```

### ตรวจสอบโมเดลที่ติดตั้ง

```bash
docker exec -it ollama ollama list
```

### ทดสอบใช้งานโมเดล

```bash
docker exec -it ollama ollama run qwen3:8b
```

หรือ

```bash
docker exec -it ollama ollama run gemma3:12b

docker exec -it ollama ollama run llama3.1:8b
```

### ทดสอบ REST API

```bash
curl http://localhost:11434/api/generate \
-d '{
  "model":"qwen3:8b",
  "prompt":"ประเทศไทยมีกี่จังหวัด",
  "stream":false
}'
```

### ตรวจสอบการใช้งาน GPU

ติดตั้งเครื่องมือ

```bash
sudo apt install radeontop -y
```

ตรวจสอบการใช้งาน GPU แบบ Real-time

```bash
radeontop
```

หรือ

```bash
watch -n 1 radeontop
```

หากค่า GPU Usage และ VRAM Usage เพิ่มขึ้นขณะรันโมเดล แสดงว่า Ollama กำลังใช้ AMD GPU ในการประมวลผล
