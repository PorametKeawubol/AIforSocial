# 🧪 Lab Sheet: การติดตั้งและใช้งาน vLLM สำหรับ LLM Serving และ RAG

# วัตถุประสงค์

1. ติดตั้งและใช้งาน vLLM  
2. เปิดใช้งาน OpenAI-Compatible API  
3. เชื่อมต่อ Python กับ vLLM  
4. สร้างระบบ RAG ด้วย FAISS  
5. วิเคราะห์ประสิทธิภาพของระบบ LLM Serving

# ตอนที่ 1 การติดตั้ง vLLM

## 1.1 ตรวจสอบเวอร์ชัน Python

รันคำสั่ง  
python \--version

### คำถาม

1. เครื่องของท่านใช้ Python เวอร์ชันใด

ตอบ ........................................................................

2. เหตุใดจึงควรใช้ Python 3.10 ขึ้นไปสำหรับงาน LLM

ตอบ ........................................................................

## 1.2 ติดตั้ง vLLM

รันคำสั่ง  
pip install vllm

จากนั้นตรวจสอบเวอร์ชัน  
python \-c "import vllm; print(vllm.\_\_version\_\_)"

### คำถาม

1. vLLM เวอร์ชันที่ติดตั้งคืออะไร

ตอบ ........................................................................

2. ระหว่างการติดตั้งพบปัญหาหรือไม่ หากพบให้อธิบาย

ตอบ ........................................................................

# ตอนที่ 2 เปิดใช้งาน LLM Server

รันคำสั่ง  
vllm serve ***Qwen/Qwen3-4B-Instruct***

ตรวจสอบโมเดล  
curl http://localhost:8000/v1/models

# **ขั้นตอนที่ 6 ทดสอบ API**

ตรวจสอบโมเดล  
curl http://localhost:8000/v1/models  
ผลลัพธ์  
{  
 "data": \[  
   {  
     "id": "Qwen/Qwen3-4B-Instruct"  
   }  
 \]  
}

### คำถาม

1. API Server ทำงานบน Port ใด

ตอบ ........................................................................

2. โมเดลที่โหลดสำเร็จคือโมเดลใด

ตอบ ........................................................................

3. หาก API Server ไม่สามารถเริ่มทำงานได้ สาเหตุอาจเกิดจากอะไรบ้าง

ตอบ ........................................................................

# ตอนที่ 3 การเชื่อมต่อด้วย Python

รันโปรแกรมตัวอย่าง

| from openai import OpenAI client \= OpenAI(     api\_key="EMPTY",     base\_url="http://localhost:8000/v1" ) response \= client.chat.completions.create(     model="Qwen/Qwen3-4B-Instruct",     messages=\[         {             "role":"user",             "content":"อธิบายความหมายของ Data Engineering"         }     \] ) print(response.choices\[0\].message.content) |
| :---- |

### คำถาม

1. ผลลัพธ์ที่โมเดลตอบคืออะไร

ตอบ ........................................................................

2. โมเดลสามารถตอบเป็นภาษาไทยได้หรือไม่

ตอบ ........................................................................

3. ทดลองเปลี่ยน Prompt เป็น

"อธิบายความแตกต่างระหว่าง Machine Learning และ Deep Learning"  
สรุปคำตอบที่ได้  
ตอบ ........................................................................

# ตอนที่ 4 Streaming Response

ทดลองรันโปรแกรมแบบ Streaming

| from openai import OpenAI client \= OpenAI(     api\_key="EMPTY",     base\_url="http://localhost:8000/v1" ) response \= client.chat.completions.create(     model="Qwen/Qwen3-4B-Instruct",     messages=\[         {"role": "user", "content": "อธิบายความหมายของ Data Engineering"}     \],     stream=True ) for chunk in response:     if chunk.choices\[0\].delta.content is not None:         print(chunk.choices\[0\].delta.content, end="", flush=True) |
| :---- |

### คำถาม

1. การตอบแบบ Streaming แตกต่างจากการตอบแบบปกติอย่างไร

ตอบ ........................................................................

2. ผู้ใช้จะได้รับประโยชน์อะไรจาก Streaming

ตอบ ........................................................................

3. ระบบ Chatbot ขนาดใหญ่ควรใช้ Streaming หรือไม่ เพราะเหตุใด

ตอบ ........................................................................

# ตอนที่ 5 การสร้างระบบ RAG

ทดลองถามคำถามจากเอกสาร

## **🎯 แนวคิด**

RAG \= Retrieval \+ Generation  
 คือ:

1. ดึงข้อมูลที่เกี่ยวข้องจากเอกสาร (Retrieval)  
2. เอาข้อมูลนั้นไปให้ LLM ตอบ (Generation)  
   ---

   # **🧰 1\) ติดตั้ง (ถ้ายังไม่ได้ติดตั้ง)**

   pip install faiss-cpu  
   pip install sentence-transformers  
   pip install numpy  
   ---

   # **📄 2\) สร้าง “เอกสารตัวอย่าง” (ข้อความง่าย ๆ)**

   documents \= \[  
      "Data Engineering คือการจัดการข้อมูลตั้งแต่การเก็บรวบรวมจนถึงการใช้งาน",  
      "Machine Learning คือการให้คอมพิวเตอร์เรียนรู้จากข้อมูล",  
      "Deep Learning คือส่วนหนึ่งของ Machine Learning ที่ใช้ Neural Network",  
      "FAISS ใช้สำหรับค้นหาความคล้ายของเวกเตอร์อย่างรวดเร็ว",  
      "RAG คือระบบที่นำข้อมูลจากเอกสารมาใช้ประกอบการตอบของ AI"  
   \]  
   ---

   # **🧠 3\) แปลงข้อความเป็น Vector (Embedding)**

   from sentence\_transformers import SentenceTransformer  
   import numpy as np  
     
   model \= SentenceTransformer("all-MiniLM-L6-v2")  
     
   embeddings \= model.encode(documents)  
   embeddings \= np.array(embeddings).astype("float32")  
   ---

   # **🗂️ 4\) สร้าง FAISS Index**

   import faiss  
     
   dimension \= embeddings.shape\[1\]  
     
   index \= faiss.IndexFlatL2(dimension)  
   index.add(embeddings)  
     
   print("จำนวนเอกสาร:", index.ntotal)  
   ---

   # **🔍 5\) ฟังก์ชันค้นหา (Retrieval)**

   def search(query, k=2):  
      query\_vec \= model.encode(\[query\]).astype("float32")  
      distances, indices \= index.search(query\_vec, k)  
     
      results \= \[\]  
      for i in indices\[0\]:  
          results.append(documents\[i\])  
     
      return results  
   ---

   # **❓ 6\) ทดลองถามคำถาม (RAG Retrieval)**

   question \= "FAISS คืออะไร"  
     
   results \= search(question)  
     
   print("🔎 ข้อมูลที่ค้นพบ:")  
   for r in results:  
      print("-", r)  
   ---

   # **🤖 7\) เอาข้อมูลไปให้ AI ตอบ (จำลอง RAG)**

   context \= "\\n".join(results)  
     
   prompt \= f"""  
   ใช้ข้อมูลต่อไปนี้ตอบคำถาม:  
     
   {context}  
     
   คำถาม: {question}  
   ตอบ:  
   """  
     
   print(prompt)

### คำถาม

1. ผลลัพธ์ที่ได้จาก RAG แตกต่างจากการถาม LLM โดยตรงอย่างไร

ตอบ ........................................................................

2. RAG ช่วยลด Hallucination ได้อย่างไร

ตอบ ........................................................................

3. หากไม่มีข้อมูลในเอกสาร ระบบควรตอบอย่างไรจึงจะเหมาะสม

ตอบ ........................................................................

# ตอนที่ 6 การตรวจสอบทรัพยากร GPU

รันคำสั่ง  
nvidia-smi

### บันทึกผล

GPU Model ........................................................................  
GPU Memory Used ........................................................................  
GPU Utilization ........................................................................

### คำถาม

1. ขณะรันโมเดล GPU ถูกใช้งานกี่เปอร์เซ็นต์

ตอบ ........................................................................

2. หาก GPU Memory เต็มจะเกิดปัญหาใด

ตอบ ........................................................................

3. เหตุใดจึงควรตั้งค่า gpu-memory-utilization ไม่เกิน 0.90

ตอบ ........................................................................

***เพิ่มเติม***ทางเทคนิค Docker Desktop บน Windows เป็นวิธีที่แนะนำที่สุด หากต้องการใช้ vLLM บน Windows โดยเฉพาะเมื่อมี NVIDIA GPU เนื่องจาก vLLM ไม่รองรับ Windows แบบ Native อย่างเป็นทางการ

# **Architecture**

Windows 11  
     │  
Docker Desktop  
     │  
WSL2 Backend  
     │  
NVIDIA GPU (CUDA)  
     │  
vLLM Container  
     │  
OpenAI Compatible API  
---

# **Step 1 ตรวจสอบ Docker**

docker \--version  
docker compose version  
---

# **Step 2 ตรวจสอบ GPU**

ใน Windows  
nvidia-smi  
ตัวอย่าง  
\+------------------------------------------------------+  
| NVIDIA-SMI 580.xx                                    |  
| RTX 4080                                             |  
\+------------------------------------------------------+  
---

# **Step 3 ทดสอบ Docker เห็น GPU**

***docker run \--rm \--gpus all nvidia/cuda:12.9.0-runtime-ubuntu24.04 nvidia-smi***  
ถ้าเห็นผลลัพธ์ของ `nvidia-smi` ภายใน Container แสดงว่า Docker พร้อมใช้ GPU แล้ว  
---

# **Step 4 Pull vLLM Image**

docker pull vllm/vllm-openai:latest  
---

# **Step 5 Run vLLM**

ตัวอย่างใช้ Qwen3 8B  
PowerShell  
docker run \`  
   \--gpus all \`  
   \-p 8000:8000 \`  
   \-v hf\_cache:/root/.cache/huggingface \`  
   \-e HUGGING\_FACE\_HUB\_TOKEN=$env:HF\_TOKEN \`  
   vllm/vllm-openai:latest \`  
   \--model Qwen/Qwen3-8B  
หากใช้โมเดลที่เป็นสาธารณะ (public) หลายรุ่น อาจไม่จำเป็นต้องใช้ `HUGGING_FACE_HUB_TOKEN` แต่โมเดลที่มีข้อกำหนดการเข้าถึงหรือ gated model จะต้องใช้  
หรือ   
docker run \`  
  \--gpus all \`  
  \-p 8000:8000 \`  
  \-v hf\_cache:/root/.cache/huggingface \`  
  vllm/vllm-openai:latest \`  
  \--model Qwen/Qwen2.5-1.5B-Instruct

---

# **Step 6 ทดสอบ API**

เปิด  
http://localhost:8000/docs  
หรือ  
http://localhost:8000/v1/models  
---

## **ทดสอบด้วย curl**

curl http://localhost:8000/v1/chat/completions ^  
\-H "Content-Type: application/json" ^  
\-d ^  
"{\\"model\\":\\"Qwen/Qwen3-8B\\",\\"messages\\":\[{\\"role\\":\\"user\\",\\"content\\":\\"Hello\\"}\]}"  
---

## **ทดสอบด้วย Python**

from openai import OpenAI

client \= OpenAI(  
   api\_key="EMPTY",  
   base\_url="http://localhost:8000/v1"  
)

response \= client.chat.completions.create(  
   model="Qwen/Qwen3-8B",  
   messages=\[  
       {"role": "user", "content": "ประเทศไทยมีกี่จังหวัด"}  
   \]  
)

print(response.choices\[0\].message.content)  
---

# **ตรวจสอบว่าใช้ GPU**

อีก Terminal  
watch \-n 1 nvidia-smi  
หรือบน Windows  
nvidia-smi \-l 1  
จะเห็น VRAM ถูกใช้งาน เช่น  
GPU Memory Usage

0 MiB

↓

8200 MiB  
