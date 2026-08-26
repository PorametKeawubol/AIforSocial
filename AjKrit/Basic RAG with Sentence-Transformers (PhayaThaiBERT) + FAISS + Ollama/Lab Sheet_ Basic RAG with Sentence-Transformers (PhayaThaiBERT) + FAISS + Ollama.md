# **🧪 Lab Sheet: Basic RAG with Sentence-Transformers (BERT) \+ FAISS \+ Ollama**

## **🎯 วัตถุประสงค์ (Objectives)**

* ทำความเข้าใจหลักการของ **RAG (Retrieval-Augmented Generation)**

* เรียนรู้การใช้ **Sentence-Transformers (PhayaThaiBERT)** เพื่อสร้าง **vector embeddings**

* ใช้ **FAISS** ในการจัดเก็บและค้นคืนเอกสาร (Vector Search)

* รวมผลลัพธ์เข้ากับ **Ollama LLM** เพื่อให้ LLM ใช้ข้อมูลจาก Knowledge Base

---

## **🔧 ขั้นตอนการทดลอง**

### **1\) ติดตั้ง Dependencies**

`pip install sentence-transformers faiss-cpu requests`

---

### **2\) เตรียมโมเดล Embedding (**`Wangchan`**BERT)**

`from sentence_transformers import SentenceTransformer`

`# โหลดโมเดล Embedding ภาษาไทย`  
`embed_model = SentenceTransformer("airesearch/wangchanberta-base-att-spm-uncased")`

---

### **3\) สร้าง Knowledge Base (ข้อมูลที่เราจะค้นหา)**

`documents = [`  
    `"โรงพยาบาลสงขลานครินทร์เป็นโรงพยาบาลศูนย์ในภาคใต้",`  
    `"ยาปฏิชีวนะควรใช้ตามคำสั่งแพทย์เพื่อป้องกันการดื้อยา",`  
    `"การออกกำลังกายช่วยลดความเสี่ยงโรคหัวใจและหลอดเลือด",`  
    `"ประเทศไทยมีการพัฒนา AI สำหรับงานด้านสาธารณสุข",`  
`]`

---

### **4\) สร้าง Embeddings \+ FAISS Index**

`import faiss`  
`import numpy as np`

`# แปลงข้อความเป็นเวกเตอร์`  
`doc_embeddings = embed_model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)`

`# สร้าง FAISS index (ใช้ Cosine Similarity)`  
`dim = doc_embeddings.shape[1]`  
`index = faiss.IndexFlatIP(dim)`  
`index.add(doc_embeddings)`

---

### **5\) ฟังก์ชันค้นคืน (Retriever)**

`def retrieve(query, top_k=2):`  
    `query_vec = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)`  
    `scores, idx = index.search(query_vec, top_k)`  
    `retrieved_docs = [documents[i] for i in idx[0]]`  
    `return retrieved_docs`

---

### **6\) ส่งข้อมูลไปยัง Ollama**

`import requests`  
`import json`

`def ask_ollama(context, question, model="qwen2.5:3b"):`  
    `prompt = f"""You are a helpful assistant. Use the following context to answer the question.`  
      
`Context:`  
`{context}`

`Question: {question}`

`Answer:"""`

    `response = requests.post(`  
        `"http://localhost:11434/api/chat",`  
        `json={`  
            `"model": model,`  
            `"messages": [{"role": "user", "content": prompt}],`  
            `"stream": False`  
        `}`  
    `)`  
    `data = response.json()`  
    `return data["message"]["content"]`

---

### **7\) รวมเป็น RAG Pipeline**

`def rag_pipeline(question):`  
    `context_docs = retrieve(question, top_k=2)`  
    `context = "\n".join(context_docs)`  
    `answer = ask_ollama(context, question)`  
    `return answer`

---

### **8\) ทดลองรันระบบ RAG**

`if __name__ == "__main__":`  
    `query = "โรงพยาบาลที่ใหญ่ที่สุดในภาคใต้คือที่ไหน?"`  
    `result = rag_pipeline(query)`  
    `print("Q:", query)`  
    `print("A:", result)`

---

## **📌 แบบฝึกหัด (Exercises)**

1. ลองเพิ่ม **เอกสารใหม่** ลงไปใน `documents` แล้วทดสอบระบบ RAG  
2. ลองใช้โมเดล Ollama ตัวอื่น เช่น `mistral`, `qwen2.5`, `llama3:instruct เลือกมาทดสอบ 3 ตัว`

Full Code

| \# ✅ Basic RAG with Sentence-Transformers (PhayaThaiBERT) \+ FAISS \+ Ollama \# install dependencies first: \# pip install sentence-transformers faiss-cpu requests  from sentence\_transformers import SentenceTransformer import faiss import numpy as np import requests import json  \# \------------------------------- \# 1\. Load embedding model (PhayaThaiBERT) \# \------------------------------- \# Huggingface: https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased \# หรือถ้าเป็น payathai-bert ให้ใช้ path ตาม repo embed\_model \= SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2") \# \------------------------------- \# 2\. In-memory knowledge base (array) \# \------------------------------- documents \= \[     "โรงพยาบาลสงขลานครินทร์เป็นโรงพยาบาลศูนย์ในภาคใต้",     "ยาปฏิชีวนะควรใช้ตามคำสั่งแพทย์เพื่อป้องกันการดื้อยา",     "การออกกำลังกายช่วยลดความเสี่ยงโรคหัวใจและหลอดเลือด",     "ประเทศไทยมีการพัฒนา AI สำหรับงานด้านสาธารณสุข", \] \# \------------------------------- \# 3\. Create embeddings and FAISS index \# \------------------------------- doc\_embeddings \= embed\_model.encode(documents, convert\_to\_numpy\=True, normalize\_embeddings\=True) dim \= doc\_embeddings.shape\[1\] index \= faiss.IndexFlatIP(dim)   \# cosine similarity (with normalized vectors) index.add(doc\_embeddings) \# \------------------------------- \# 4\. Define retrieval function \# \------------------------------- def retrieve(query, top\_k\=2):     query\_vec \= embed\_model.encode(\[query\], convert\_to\_numpy\=True, normalize\_embeddings\=True)     scores, idx \= index.search(query\_vec, top\_k)     retrieved\_docs \= \[documents\[i\] for i in idx\[0\]\]     return retrieved\_docs  \# \------------------------------- \# 5\. Query Ollama for answer \# \------------------------------- def ask\_ollama(context, question):     import requests     url \= "http://localhost:11434/api/chat"     payload \= {         "model": "`qwen2.5:3b`",  \# เปลี่ยนเป็นโมเดลที่คุณ pull มา         "messages": \[             {"role": "system", "content": "You are a helpful assistant."},             {"role": "user", "content": f"Context:\\n{context}\\n\\nQuestion:\\n{question}"}         \],         "stream": False     }     response \= requests.post(url, json\=payload)     if response.status\_code \!= 200:         raise RuntimeError(f"Ollama API error: {response.status\_code}, {response.text}")     data \= response.json()     \# บางเวอร์ชันของ Ollama อาจใช้ key อื่น เช่น 'response'     if "message" in data and "content" in data\["message"\]:         return data\["message"\]\["content"\]     elif "response" in data:           return data\["response"\]  \# fallback     else:         raise KeyError(f"Unexpected response format: {data}") \# \------------------------------- \# 6\. Full RAG pipeline \# \------------------------------- def rag\_pipeline(question):     context\_docs \= retrieve(question, top\_k\=2)     context \= "\\n".join(context\_docs)     answer \= ask\_ollama(context, question)     return answer  \# \------------------------------- \# 7\. Example run \# \------------------------------- if \_\_name\_\_ \== "\_\_main\_\_":     query \= "โรงพยาบาลที่ใหญ่ที่สุดในภาคใต้คือที่ไหน?"     result \= rag\_pipeline(query)     print("Q:", query)     print("A:", result)  |
| :---- |

