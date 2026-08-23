# คำตอบ Lab vLLM Serving และ RAG บน AMD ROCm

## สรุปการปรับจาก lab เดิม

- เครื่องนี้ไม่มีคำสั่ง `python` แต่ใช้ `python3` ได้
- ใช้ AMD GPU จึงเปลี่ยนจาก `nvidia-smi` เป็น `rocm-smi`
- ติดตั้ง vLLM ROCm แยกใน `.venv-vllm` ด้วย wheel จาก `https://wheels.vllm.ai/rocm/`
- ใช้โมเดล `Qwen/Qwen3-0.6B` แทน `Qwen/Qwen3-4B-Instruct` เพื่อให้เหมาะกับ VRAM 8 GB
- Docker Desktop ใช้กับ ROCm container ไม่ได้ในเครื่องนี้ เพราะ daemon อยู่ใน LinuxKit VM และไม่เห็น `/dev/kfd`/`/dev/dri` จึงใช้ host-native vLLM ROCm แทน

## ตอนที่ 1 การติดตั้ง vLLM

1. เครื่องใช้ Python เวอร์ชันใด

ตอบ: Python 3.12.3 (`python3 --version`)

2. เหตุใดจึงควรใช้ Python 3.10 ขึ้นไปสำหรับงาน LLM

ตอบ: ไลบรารี LLM รุ่นใหม่ เช่น PyTorch, Transformers, vLLM และ OpenAI SDK มักรองรับและทดสอบกับ Python รุ่นใหม่มากกว่า ทำให้ติดตั้ง dependency ได้ง่ายและลดปัญหา compatibility

3. vLLM เวอร์ชันที่ติดตั้งคืออะไร

ตอบ: `vLLM 0.24.0+rocm723`

4. ระหว่างการติดตั้งพบปัญหาหรือไม่

ตอบ: พบปัญหา 2 จุด

- `pip install vllm` จาก PyPI ปกติดึง dependency ฝั่ง CUDA ไม่เหมาะกับ AMD GPU
- Docker Desktop รัน ROCm container ไม่ได้เพราะไม่เห็น `/dev/kfd` และ `/dev/dri`

วิธีแก้คือใช้ vLLM ROCm wheel และเติม runtime libraries ที่ขาดแบบ local ใน `W1D4/vllm-rocm-lab/system-libs`

## ตอนที่ 2 เปิดใช้งาน LLM Server

คำสั่งที่ใช้:

```bash
cd W1D4/vllm-rocm-lab
./run_vllm_rocm_host.sh
curl http://127.0.0.1:8000/v1/models
```

1. API Server ทำงานบน Port ใด

ตอบ: 8000

2. โมเดลที่โหลดสำเร็จคือโมเดลใด

ตอบ: `Qwen/Qwen3-0.6B`

3. หาก API Server ไม่สามารถเริ่มทำงานได้ สาเหตุอาจเกิดจากอะไรบ้าง

ตอบ: ROCm runtime library ไม่ครบ, VRAM ไม่พอ, โมเดลดาวน์โหลดไม่สำเร็จ, พอร์ต 8000 ถูกใช้อยู่, Docker ไม่เห็น `/dev/kfd`/`/dev/dri` หรือการ์ด AMD ต้องตั้ง `HSA_OVERRIDE_GFX_VERSION`

## ตอนที่ 3 การเชื่อมต่อด้วย Python

ใช้ไฟล์:

```bash
.venv/bin/python W1D4/vllm-rocm-lab/client_chat.py
```

1. ผลลัพธ์ที่โมเดลตอบคืออะไร

ตอบ: โมเดลตอบว่า Data Engineering คือการจัดการข้อมูล เช่น การรวบรวม จัดเก็บ ประมวลผล และเตรียมข้อมูลให้ใช้งานได้อย่างมีประสิทธิภาพในองค์กร

2. โมเดลสามารถตอบเป็นภาษาไทยได้หรือไม่

ตอบ: ได้

3. ทดลองเปลี่ยน Prompt เป็น "อธิบายความแตกต่างระหว่าง Machine Learning และ Deep Learning" สรุปคำตอบที่ได้

ตอบ: Machine Learning คือการให้คอมพิวเตอร์เรียนรู้รูปแบบจากข้อมูล ส่วน Deep Learning เป็นแขนงหนึ่งของ Machine Learning ที่ใช้โครงข่ายประสาทเทียมหลายชั้น เหมาะกับข้อมูลซับซ้อน เช่น ภาพ เสียง และภาษา

## ตอนที่ 4 Streaming Response

ใช้ไฟล์:

```bash
.venv/bin/python W1D4/vllm-rocm-lab/client_stream.py
```

1. การตอบแบบ Streaming แตกต่างจากการตอบแบบปกติอย่างไร

ตอบ: แบบปกติรอให้โมเดลสร้างคำตอบครบก่อนจึงส่งกลับ ส่วน Streaming ส่งคำตอบออกมาทีละส่วนระหว่างที่โมเดลกำลัง generate

2. ผู้ใช้จะได้รับประโยชน์อะไรจาก Streaming

ตอบ: เห็นคำตอบเร็วขึ้น ลดความรู้สึกรอนาน และเหมาะกับคำตอบยาวหรือระบบแชต

3. ระบบ Chatbot ขนาดใหญ่ควรใช้ Streaming หรือไม่ เพราะเหตุใด

ตอบ: ควรใช้ เพราะช่วยลด perceived latency และทำให้ประสบการณ์ใช้งานเป็นธรรมชาติมากขึ้น

## ตอนที่ 5 การสร้างระบบ RAG

ใช้ไฟล์:

```bash
.venv/bin/python W1D4/vllm-rocm-lab/rag_faiss.py
```

ผลที่รันได้จริง:

```text
จำนวนเอกสาร: 5
ข้อมูลที่ค้นพบ:
- FAISS ใช้สำหรับค้นหาความคล้ายของเวกเตอร์อย่างรวดเร็ว
- RAG คือระบบที่นำข้อมูลจากเอกสารมาใช้ประกอบการตอบของ AI

คำตอบจาก RAG:
<think>

</think>

FAISS คือระบบที่ใช้สำหรับค้นหาความคล้ายของเวกเตอร์อย่างรวดเร็ว
```

1. ผลลัพธ์ที่ได้จาก RAG แตกต่างจากการถาม LLM โดยตรงอย่างไร

ตอบ: RAG จะดึงเอกสารที่เกี่ยวข้องมาเป็น context ก่อนตอบ ทำให้คำตอบอ้างอิงข้อมูลที่กำหนดไว้ ไม่ได้พึ่งความรู้ภายในโมเดลอย่างเดียว

2. RAG ช่วยลด Hallucination ได้อย่างไร

ตอบ: RAG จำกัดคำตอบให้ใช้ข้อมูลจากเอกสารที่ค้นคืนมา จึงลดโอกาสที่โมเดลจะเดาหรือสร้างข้อมูลที่ไม่มีแหล่งอ้างอิง

3. หากไม่มีข้อมูลในเอกสาร ระบบควรตอบอย่างไรจึงจะเหมาะสม

ตอบ: ควรตอบว่าไม่พบข้อมูลในเอกสาร หรือข้อมูลไม่เพียงพอ ไม่ควรแต่งคำตอบเอง

## ตอนที่ 6 การตรวจสอบทรัพยากร GPU

คำสั่งที่ใช้แทน `nvidia-smi`:

```bash
rocm-smi --showproductname --showmeminfo vram --showuse
```

GPU Model: Navi 33 [Radeon RX 7700S/7600/7600S/7600M XT/PRO W7600]

GPU Memory Used ระหว่างรัน vLLM: 8275853312 bytes จาก 8573157376 bytes
