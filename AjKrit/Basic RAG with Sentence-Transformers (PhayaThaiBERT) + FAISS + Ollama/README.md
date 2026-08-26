# Basic RAG: Sentence-Transformers + FAISS + Ollama

โปรเจกต์นี้ทำตาม Full Code ในใบงาน: แปลง 4 เอกสารเป็น normalized embeddings, ค้นด้วย FAISS `IndexFlatIP` และส่ง context ที่ค้นได้ไปให้ Ollama ตอบคำถาม

ค่าเริ่มต้นของ embedding model คือ `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` ตาม Full Code ในใบงาน (แม้ชื่อใบงานจะกล่าวถึง PhayaThaiBERT)

## โมเดลที่ทดสอบทีละตัว

| ระดับ | โมเดล | เหตุผล |
| --- | --- | --- |
| เล็ก | `qwen2.5:1.5b` | รองรับหลายภาษา ใช้ VRAM น้อย |
| กลาง | `llama3.2:3b` | ขนาดพอดีกับงานสนทนาและเครื่อง 8GB VRAM |
| ใหญ่ | `qwen2.5:7b` | คุณภาพภาษาไทย/การทำตามคำสั่งสูงขึ้น และยังพอใส่ RX 7600S 8GB ได้ |

Ollama จะโหลดและทดสอบเพียงโมเดลเดียวต่อคำสั่ง โดย script ส่ง `keep_alive: 0` เพื่อคืน VRAM หลังจบคำตอบ

## รัน

เปิด Terminal หนึ่งหน้าต่างสำหรับ Ollama:

```bash
cd "/home/porametk/Desktop/AIforSocial/AIforSocial/AjKrit/Basic RAG with Sentence-Transformers (PhayaThaiBERT) + FAISS + Ollama"
./run_ollama_rocm.sh
```

เปิด Terminal ใหม่ แล้วทดสอบ **ทีละตัว** (เริ่มจากตัวเล็ก):

```bash
cd "/home/porametk/Desktop/AIforSocial/AIforSocial/AjKrit/Basic RAG with Sentence-Transformers (PhayaThaiBERT) + FAISS + Ollama"
rag_python="../Vector Search ระหว่าง FAISS และ ChromaDB/.venv-gfx1102/bin/python"
"$rag_python" basic_rag_ollama.py --model qwen2.5:1.5b --pull-missing
```

เมื่อผลตัวแรกเสร็จแล้ว จึงค่อยรันตัวกลางและตัวใหญ่:

```bash
"$rag_python" basic_rag_ollama.py --model llama3.2:3b --pull-missing
"$rag_python" basic_rag_ollama.py --model qwen2.5:7b --pull-missing
```

ผลลัพธ์ของแต่ละโมเดลจะอยู่ใน `results/<ชื่อโมเดล>.json` พร้อม context ที่ค้นได้ คำตอบ และเวลา generate

หากต้องการใช้ WangchanBERT ตามหัวข้อใบงานแทน MiniLM ให้เพิ่ม:

```bash
--embedding-model airesearch/wangchanberta-base-att-spm-uncased
```
