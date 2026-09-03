# **🧪 ใบงานปฏิบัติ: Low-VRAM LLM Engineering**

**หัวข้อ:** การสร้างและใช้งาน Micro-LLM บนเครื่อง GPU VRAM 2GB  
**ระดับ:** ปริญญาโท  
**เวลา:** 3 ชั่วโมง  
**จุดเริ่มต้น:** ขั้นตอนที่ 5  
**Prerequisite:** เครื่องติดตั้ง Ollama และ Python แล้ว

---

## **🎯 วัตถุประสงค์**

เมื่อจบ Lab นักศึกษาสามารถ

* รัน Micro-LLM ด้วย Ollama  
* เปรียบเทียบ Model ขนาดเล็ก  
* ตรวจสอบการใช้ VRAM ขณะ LLM ทำงาน  
* เข้าใจ CPU–GPU Offloading  
* เรียก LLM ผ่าน LangChain  
* สร้าง Local RAG  
* วิเคราะห์ผลกระทบของ Context Size  
* ออกแบบระบบ Low-VRAM LLM

---

# **Part 1 — ทดลอง Micro-LLM**

## **ขั้นที่ 5: ดาวน์โหลดและรัน Model**

เริ่มจาก Model ขนาดเล็ก

### **5.1 Qwen2.5 0.5B**

เปิด Terminal

ollama pull qwen2.5:0.5b

ตรวจสอบ Model

ollama list

รัน Model

ollama run qwen2.5:0.5b

ทดลองถาม

What is a GPU?

ทดลองภาษาไทย

GPU คืออะไร และทำหน้าที่อะไร?

ออกจาก Model

/bye  
---

## **5.2 Llama 3.2 1B**

ollama pull llama3.2:1b

รัน

ollama run llama3.2:1b

ทดลอง

Explain HTTP/3 in simple terms.

ทดลองภาษาไทย

อธิบายความแตกต่างระหว่าง HTTP/2 และ HTTP/3  
---

## **5.3 SmolLM2**

ทดลอง Model ขนาดใหญ่ขึ้น

ollama pull smollm2:1.7b

รัน

ollama run smollm2:1.7b  
---

# **📝 Activity 1 — เปรียบเทียบ Micro-LLM**

ใช้คำถามเดียวกันกับทุก Model

Explain the difference between HTTP/2 and HTTP/3.

บันทึกผล

| Model | Parameter | Response Quality | Speed | ความเห็น |
| ----- | ----- | ----- | ----- | ----- |
| Qwen2.5 | 0.5B |  |  |  |
| Llama 3.2 | 1B |  |  |  |
| SmolLM2 | 1.7B |  |  |  |

### **คำถาม**

**Q1.** Model ใดตอบเร็วที่สุด?

**Q2.** Model ใดให้คำตอบดีที่สุด?

**Q3.** Model ขนาดใหญ่ขึ้นให้ผลดีขึ้นเสมอหรือไม่?

---

# **Part 2 — ตรวจสอบ GPU Offloading**

## **ขั้นที่ 6: Monitor VRAM**

เปิด Terminal ใหม่

### **Windows NVIDIA GPU**

nvidia-smi \-l 1

คำสั่งนี้จะ update การใช้ GPU ทุก 1 วินาที

ตัวอย่างข้อมูลที่ต้องสังเกต

GPU Memory Usage  
\----------------  
ก่อนรัน Model  
ขณะรัน Model  
หลัง /bye

จาก Terminal อีกหน้าต่าง

ollama run llama3.2:1b

ส่งคำถาม

Explain Retrieval-Augmented Generation.

กลับไปดู `nvidia-smi`

---

## **📝 Activity 2 — VRAM Measurement**

| สถานะ | VRAM Used |
| ----- | ----- |
| ก่อน Run LLM |  |
| ขณะ Model Loaded |  |
| ขณะ Generate |  |
| หลัง `/bye` |  |

### **คำถาม**

**Q4.** การ Run LLM ทำให้ VRAM เพิ่มขึ้นหรือไม่?

**Q5.** Model ใช้ VRAM ทั้งหมดหรือไม่?

**Q6.** ถ้า Model มีขนาดใหญ่กว่า VRAM จะเกิดอะไรขึ้น?

---

# **Part 3 — LangChain \+ Ollama**

## **ขั้นที่ 8: สร้าง Python Environment**

สร้าง Project

mkdir low-vram-llm  
cd low-vram-llm

สร้าง Virtual Environment

python \-m venv venv

Windows

venv\\Scripts\\activate

ติดตั้ง

pip install langchain langchain-ollama  
---

# **ขั้นที่ 9: สร้าง `app.py`**

from langchain\_ollama import ChatOllama

llm \= ChatOllama(  
    model="llama3.2:1b",  
    num\_gpu=99,  
    num\_ctx=1024,  
    num\_thread=4  
)

response \= llm.invoke(  
    "Explain HTTP/3 in simple terms."  
)

print(response.content)

Run

python [app.py](http://app.py)   
Measure-Command { python [app.py](http://app.py) | Out-Host }

---

# **📝 Activity 4 — ทดลอง Parameter**

ทดลอง

num\_ctx=512

จากนั้น

num\_ctx=1024

และ

num\_ctx=2048

บันทึกผล

| `num_ctx` | Response Time | VRAM | Quality |
| ----- | ----- | ----- | ----- |
| 512 |  |  |  |
| 1024 |  |  |  |
| 2048 |  |  |  |

### **คำถาม**

**Q11.** เมื่อเพิ่ม Context Size เกิดอะไรขึ้นกับ Memory?

**Q12.** สำหรับ GPU 2GB ควรใช้ Context Size ประมาณเท่าใด?

---

# **Part 5 — สร้าง Local RAG**

## **ขั้นที่ 10: สร้าง Knowledge Base**

สร้างไฟล์

knowledge.txt

ใส่ข้อมูล

| HTTP/2 uses TCP as its transport protocol. HTTP/2 supports multiplexing multiple streams over a single TCP connection. HTTP/3 uses QUIC as its transport protocol. QUIC is built on top of UDP. HTTP/3 can reduce the impact of head-of-line blocking. QUIC provides encrypted transport and supports faster connection establishment. |
| :---- |

---

# **ขั้นที่ 11: ติดตั้ง RAG Libraries**

pip install chromadb sentence-transformers  
---

# **ขั้นที่ 12: สร้าง RAG Pipeline**

Architecture

            knowledge.txt  
                   │  
                   ▼  
             Text Splitting  
                   │  
                   ▼  
               Embedding  
                   │  
                   ▼  
               ChromaDB  
                   │  
              Similarity  
                Search  
                   │  
                   ▼  
               Top-K Docs  
                   │  
                   ▼  
             Llama 3.2 1B  
                   │  
                   ▼  
                 Answer

กำหนด

RETRIEVAL\_K \= 3

| import os import chromadb from chromadb.utils import embedding\_functions import ollama \# \--- ขั้นที่ 10: เตรียมไฟล์ knowledge.txt \--- KNOWLEDGE\_FILE \= "knowledge.txt" KNOWLEDGE\_CONTENT \= """HTTP/2 uses TCP as its transport protocol. HTTP/2 supports multiplexing multiple streams over a single TCP connection. HTTP/3 uses QUIC as its transport protocol. QUIC is built on top of UDP. HTTP/3 can reduce the impact of head-of-line blocking. QUIC provides encrypted transport and supports faster connection establishment.""" \# สร้างไฟล์หากยังไม่มี if not os.path.exists(KNOWLEDGE\_FILE):     with open(KNOWLEDGE\_FILE, "w", encoding="utf-8") as f:         f.write(KNOWLEDGE\_CONTENT) \# \--- ขั้นที่ 12: RAG Pipeline \--- \# 1\. Text Splitting: อ่านและแบ่งข้อมูลตามบรรทัดว่าง with open(KNOWLEDGE\_FILE, "r", encoding="utf-8") as f:     raw\_text \= f.read() documents \= \[doc.strip() for doc in raw\_text.split("\\n\\n") if doc.strip()\] \# 2\. Embedding: ใช้ Sentence-Transformers (all-MiniLM-L6-v2) embedding\_func \= embedding\_functions.SentenceTransformerEmbeddingFunction(     model\_name="all-MiniLM-L6-v2" ) \# 3\. ChromaDB: สร้าง Vector Database ในหน่วยความจำ (In-Memory) chroma\_client \= chromadb.Client() collection \= chroma\_client.create\_collection(     name="network\_knowledge",      embedding\_function=embedding\_func ) \# บันทึกข้อมูลลง ChromaDB ids \= \[f"doc\_{i}" for i in range(len(documents))\] collection.add(documents=documents, ids=ids) \# 4\. Similarity Search & LLM Generation RETRIEVAL\_K \= 3 def ask\_rag(query: str):     \# Similarity Search ดึง Top-K Docs     results \= collection.query(         query\_texts=\[query\],         n\_results=RETRIEVAL\_K     )          retrieved\_docs \= results\["documents"\]\[0\]     context \= "\\n- ".join(retrieved\_docs)          \# สร้าง Prompt สำหรับ Llama 3.2 1B     prompt \= f"""Answer the question based only on the provided context. Context: \- {context} Question: {query} Answer:"""     \# ส่งเข้า Llama 3.2 1B ผ่าน Ollama     response \= ollama.generate(         model="llama3.2:1b",         prompt=prompt     )          return response\["response"\], retrieved\_docs \# \--- ทดสอบการใช้งาน \--- if \_\_name\_\_ \== "\_\_main\_\_":     user\_question \= "What transport protocol does HTTP/3 use?"          answer, docs \= ask\_rag(user\_question)          print(f"Question: {user\_question}\\n")     print("--- Top-K Retrieved Docs \---")     for i, doc in enumerate(docs, 1):         print(f"{i}. {doc}")              print("\\n--- Llama 3.2 1B Answer \---")     print(answer)  |
| :---- |

---

# **📝 Activity 5 — ทดลอง Local RAG**

ทดลองคำถาม

HTTP/3 ใช้ Transport Protocol อะไร?  
HTTP/2 และ HTTP/3 แตกต่างกันอย่างไร?  
QUIC มีความเกี่ยวข้องกับ HTTP/3 อย่างไร?

บันทึก

| Question | Retrieved Docs | Answer Correct? |
| ----- | ----- | ----- |
| HTTP/3 ใช้อะไร? |  |  |
| HTTP/2 vs HTTP/3 |  |  |
| QUIC คืออะไร? |  |  |

---

# **Part 6 — LLM vs RAG**

## **ขั้นที่ 13: เปรียบเทียบ**

### **Case A — LLM อย่างเดียว**

Question  
   ↓  
Llama 3.2 1B  
   ↓  
Answer

### **Case B — RAG**

Question  
   ↓  
ChromaDB  
   ↓  
Relevant Documents  
   ↓  
Llama 3.2 1B  
   ↓  
Answer  
---

## **📝 Activity 6**

ทดลองคำถาม

According to the knowledge base,  
what transport protocol does HTTP/3 use?

เปรียบเทียบ

| Metric | LLM | RAG |
| ----- | ----- | ----- |
| Correctness |  |  |
| Response Time |  |  |
| Specific Knowledge |  |  |
| Hallucination |  |  |

### **คำถาม**

**Q13.** ทำไม RAG จึงช่วยให้ Model ขนาด 1B ตอบคำถามเฉพาะ Domain ได้ดีขึ้น?

---

# **Part 7 — Engineering Challenge** 

## **ขั้นที่ 14: สร้าง Low-VRAM AI Assistant**

ให้นักศึกษาออกแบบระบบ

                    User  
                       │  
                       ▼  
                 Query Router  
                       │  
              ┌────────┴────────┐  
              │                 │  
              ▼                 ▼  
         Simple Query      Knowledge Query  
              │                 │  
              ▼                 ▼  
        Qwen 0.5B           ChromaDB  
                                │  
                                ▼  
                             Top-K  
                                │  
                                ▼  
                           Llama 1B  
                                │  
                                ▼  
                             Answer

ollama pull qwen2.5:0.5b  
ollama pull llama3.2:1b

routed\_rag.py 

| import chromadb from chromadb.utils import embedding\_functions import ollama \# \========================================== \# 1\. SETUP KNOWLEDGE BASE (ChromaDB) \# \========================================== embedding\_func \= embedding\_functions.SentenceTransformerEmbeddingFunction(     model\_name="all-MiniLM-L6-v2" ) chroma\_client \= chromadb.Client() collection \= chroma\_client.create\_collection(     name="network\_docs",      embedding\_function=embedding\_func ) \# Populate vector store documents \= \[     "HTTP/2 uses TCP as its transport protocol.",     "HTTP/2 supports multiplexing multiple streams over a single TCP connection.",     "HTTP/3 uses QUIC as its transport protocol.",     "QUIC is built on top of UDP.",     "HTTP/3 can reduce the impact of head-of-line blocking.",     "QUIC provides encrypted transport and supports faster connection establishment." \] collection.add(     documents=documents,      ids=\[f"doc\_{i}" for i in range(len(documents))\] ) \# \========================================== \# 2\. QUERY ROUTER \# \========================================== def route\_query(user\_query: str) \-\> str:     """Uses Qwen 0.5B to classify the query type."""     router\_prompt \= f"""Classify the user input into exactly one category: 'SIMPLE' or 'KNOWLEDGE'. \- SIMPLE: Greetings, small talk, general coding, math, or common knowledge. \- KNOWLEDGE: Specific questions about networking protocols (HTTP/2, HTTP/3, QUIC, TCP, UDP). User Input: "{user\_query}" Category (Reply with ONLY 'SIMPLE' or 'KNOWLEDGE'):"""     res \= ollama.generate(model="qwen2.5:0.5b", prompt=router\_prompt)     category \= res\["response"\].strip().upper()          return "KNOWLEDGE" if "KNOWLEDGE" in category else "SIMPLE" \# \========================================== \# 3\. BRANCH HANDLERS \# \========================================== def handle\_simple\_query(query: str) \-\> str:     """Branch A: Direct LLM response via Qwen 0.5B"""     response \= ollama.generate(model="qwen2.5:0.5b", prompt=query)     return response\["response"\] def handle\_knowledge\_query(query: str, top\_k: int \= 3\) \-\> str:     """Branch B: ChromaDB Retrieval \-\> Top-K \-\> Llama 1B Generation"""     \# Vector Similarity Search     results \= collection.query(query\_texts=\[query\], n\_results=top\_k)     retrieved\_docs \= results\["documents"\]\[0\]     context \= "\\n- ".join(retrieved\_docs)          \# RAG Prompt for Llama 1B     rag\_prompt \= f"""Answer the question based strictly on the provided context. Context: \- {context} Question: {query} Answer:"""     response \= ollama.generate(model="llama3.2:1b", prompt=rag\_prompt)     return response\["response"\] \# \========================================== \# 4\. MAIN PIPELINE \# \========================================== def process\_request(query: str):     print(f"\\n\[User Query\]: {query}")          \# Step 1: Query Router     route \= route\_query(query)     print(f"└─► Router Decision: \[{route}\]")          \# Step 2: Route Execution     if route \== "SIMPLE":         print("└─► Path: Simple Query ──► Qwen 0.5B")         answer \= handle\_simple\_query(query)     else:         print("└─► Path: Knowledge Query ──► ChromaDB Top-K ──► Llama 1B")         answer \= handle\_knowledge\_query(query)              print(f"\\n\[Response\]:\\n{answer.strip()}\\n" \+ "="\*50) \# \========================================== \# TEST EXAMPLES \# \========================================== if \_\_name\_\_ \== "\_\_main\_\_":     \# Test 1: Simple Casual Query     process\_request("Hi there\! Can you write a quick 3-line poem about coffee?")          \# Test 2: Knowledge Query     process\_request("What transport protocol does HTTP/3 use and why?")  |
| :---- |

---

## **กติกา**

### **Simple Query**

ตัวอย่าง

What is GPU?  
What is HTTP?  
What is RAG?

ใช้

Qwen2.5 0.5B

### **Knowledge Query**

ตัวอย่าง

According to our knowledge base,  
what protocol does HTTP/3 use?

ใช้

ChromaDB  
\+  
Llama 3.2 1B  
---

### **🎓 ผลลัพธ์สุดท้ายของ Lab**

ผู้เรียนควรสามารถอธิบายได้ว่า

> **"แม้ GPU มี VRAM เพียง 2GB แต่สามารถสร้าง LLM Application ได้ โดยเลือก Model ขนาดเล็ก ใช้ Quantization, CPU–GPU Offloading, RAG และ Model Routing เพื่อชดเชยข้อจำกัดด้าน Hardware"**

