## **🧪 Sheet Lab: การสร้างระบบ RAG ง่าย ๆ ด้วย Ollama \+ Python**

### **🎯 วัตถุประสงค์**

* เข้าใจหลักการของ RAG (Retrieval-Augmented Generation)
* ใช้ไฟล์ข้อมูลภายในเครื่อง (เช่น `.txt`) เป็นแหล่งค้นคืนข้อมูล
* ส่งข้อความไปยังโมเดลของ Ollama พร้อมบริบทที่ได้จากการค้นหา

---

### **🧩 ความเข้าใจเบื้องต้นเกี่ยวกับ RAG**

RAG \= การนำเอกสารที่เกี่ยวข้อง (Retrieval) มาใช้ร่วมกับ LLM ในการตอบคำถาม (Generation)
----------------------------------------------------------------------------------------------------------------------------------------

### **✅ สิ่งที่ต้องเตรียม**

#### **1\. ติดตั้ง Ollama และรันโมเดล:**

bash
คัดลอกแก้ไข
`ollama pull llama3`
`ollama run llama3`

โมเดลจะรันบน `http://localhost:11434`

#### **2\. ติดตั้ง Python Libraries:**

bash
คัดลอกแก้ไข
`pip install requests sentence-transformers faiss-cpu`

---

### **📁 เตรียมข้อมูล**

สร้างไฟล์ชื่อ `knowledge.txt`:
txt
คัดลอกแก้ไข
`ประเทศไทยมีจังหวัดทั้งหมด 77 จังหวัด โดยกรุงเทพมหานครมีสถานะเป็นทั้งจังหวัดและเขตปกครองพิเศษ`
`จังหวัดที่มีพื้นที่มากที่สุดคือ นครราชสีมา ส่วนจังหวัดที่เล็กที่สุดคือ สมุทรสงคราม`

---

### **✅ ตัวอย่าง Python Lab: RAG ง่าย ๆ**

python
คัดลอกแก้ไข
`import faiss`
`import numpy as np`
`import requests`
`from sentence_transformers import SentenceTransformer`

`# โหลดโมเดลฝังข้อความ`
`embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")`

`# โหลดฐานความรู้`
`with open("knowledge.txt", "r", encoding="utf-8") as f:`
    `knowledge = f.readlines()`

`# สร้างเวกเตอร์จากข้อมูล`
`docs_embeddings = embedder.encode(knowledge)`
`index = faiss.IndexFlatL2(docs_embeddings.shape[1])`
`index.add(np.array(docs_embeddings))`

`# รับคำถามจากผู้ใช้`
`question = input("🔎 ป้อนคำถาม: ")`

`# แปลงคำถามเป็นเวกเตอร์และค้นหา`
`q_embedding = embedder.encode([question])`
`D, I = index.search(np.array(q_embedding), k=1)`

`# ดึงเนื้อหาที่ใกล้เคียง`
`retrieved_text = knowledge[I[0][0]]`

`# สร้าง prompt รวมบริบท`
`prompt = f"จากข้อความนี้: \"{retrieved_text.strip()}\"\nตอบคำถาม: {question}"`

`# ส่งไปยัง Ollama`
`response = requests.post("http://localhost:11434/api/generate", json={`
    `"model": "llama3",`
    `"prompt": prompt,`
    `"stream": False`
`})`

`print("\n💬 คำตอบจาก LLM:", response.json()["response"])`

---

### **🧪 ตัวอย่างการใช้งาน**

**คำถาม:**
คัดลอกแก้ไข
`Quantum หรือ ควอนตัม คืออะไรอธิบายมาให้ครอบคลุมที่สุด`

**ผลลัพธ์ (LLM):**
คัดลอกแก้ไข


---

### **🎒 กิจกรรม**

* เปลี่ยนไฟล์ `knowledge.txt` เป็นบทความจากวิกิพีเดีย
* เพิ่มจำนวนบริบทจาก `k=1` เป็น `k=3` แล้วรวมข้อมูลหลายบรรทัด
* ลองเปรียบเทียบผลระหว่าง llama3 กับ mistral หรือ gemma หรือ โมเดลมราสนใจ อื่น ๆ ไม่น้อยกว่า 2 โมเดล

#### \*\*\*ส่งงานผลลัพธ์แต่ละ โมเดล พร้อมระบุชื่อโมเดล \*\*\*
