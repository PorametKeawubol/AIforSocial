# **🧪 ใบงานฝึกปฏิบัติ: การใช้งาน Vector Database (ChromaDB) เพื่อจัดเก็บและค้นหาข้อมูลสำหรับ Hybrid RAG**

### **🎯 วัตถุประสงค์ (Objectives)**

1. สามารถนำ Dense Model ภาษาไทยที่คัดเลือกได้จากใบงานเดิม มาเชื่อมต่อกับ **ChromaDB**

2. เข้าใจการจัดเก็บเอกสารพร้อม **Metadata** (เช่น รหัสเอกสาร หมวดหมู่) เพื่อรองรับระบบ RAG ในระดับ Production  
3. สามารถค้นหาข้อมูลด้วย **Semantic Search** ร่วมกับ **Metadata Filtering** เพื่อแก้ปัญหา Exact Code Match (เช่น รหัส D-9902)  
4. เข้าใจการบันทึกข้อมูลแบบถาวร (Persistent Storage) และการจัดการ Collection ใน Vector Database

### **🔧 สิ่งที่ต้องเตรียม (Prerequisites)**

ติดตั้ง Library ที่จำเป็นผ่าน Terminal/Notebook:

pip install chromadb sentence-transformers

### **📘 Step 1: นำเข้า Libraries และตั้งค่า Persistent Storage**

ในขั้นตอนนี้ เราจะใช้ PersistentClient เพื่อให้ ChromaDB บันทึก Index และ Data ลงใน Disk (แตกต่างจากใน Memory ทั่วไป)

| import chromadb from chromadb.utils import embedding\_functions \# 1\. ตั้งค่าการบันทึกข้อมูลลง Disk ในโฟลเดอร์ ./chroma\_db chroma\_client \= chromadb.PersistentClient(path="./chroma\_db")  print("ChromaDB Client initialized successfully\!") |
| :---- |

### **📘 Step 2: เชื่อมต่อ Dense Embedding Model จากใบงานเดิม**

นำโมเดลภาษาไทยที่ได้คะแนนสูงสุดจากใบงานประเมินโมเดล (เช่น BAAI/bge-m3 หรือ intfloat/multilingual-e5-large) มาสร้างเป็น Embedding Function ของ ChromaDB

| \# เลือก Model ภาษาไทยที่คัดเลือกได้จากใบงานประเมินโมเดล \# เช่น BAAI/bge-m3 หรือ sentence-transformers/paraphrase-multilingual-mpnet-base-v2 SELECTED\_MODEL\_NAME \= "BAAI/bge-m3" sentence\_transformer\_ef \= embedding\_functions.SentenceTransformerEmbeddingFunction(     model\_name=SELECTED\_MODEL\_NAME ) print(f"Loaded Embedding Function using: {SELECTED\_MODEL\_NAME}") |
| :---- |

### **📘 Step 3: สร้าง Collection และใส่ข้อมูลพร้อม Metadata**

สร้าง Collection และใส่เอกสารจากชุดทดสอบเดิมลงไป โดยเพิ่ม **Metadata** เพื่อจัดหมวดหมู่และบันทึกรหัสเอกสารเฉพาะ

| \# สร้างหรือเรียกใช้ Collection collection \= chroma\_client.get\_or\_create\_collection(     name="thai\_rag\_docs",     embedding\_function=sentence\_transformer\_ef,     metadata={"hnsw:space": "cosine"} \# กำหนด Similarity Metric เป็น Cosine ) \# ข้อมูลเอกสารและ Metadata จากชุดทดสอบเดิม documents \= \[     "นโยบายการเบิกจ่ายค่าเดินทางและค่าเบี้ยเลี้ยงต่างจังหวัดประจำปี 2024",     "คู่มือการใช้งานระบบสารสนเทศรหัสเอกสาร D-9902 สำหรับเจ้าหน้าที่ IT",     "แนวทางการขออนุญาตลาพักร้อนและการลาป่วยผ่านระบบออนไลน์" \] metadatas \= \[     {"doc\_id": "Doc 1", "category": "HR", "doc\_code": "D-1001"},     {"doc\_id": "Doc 3", "category": "IT", "doc\_code": "D-9902"},  \# เอกสารที่มีรหัสเฉพาะ     {"doc\_id": "Doc 2", "category": "HR", "doc\_code": "D-1002"} \]  ids \= \["doc\_1", "doc\_3", "doc\_2"\] \# บันทึกข้อมูลลงใน Vector Database collection.add(     documents=documents,     metadatas=metadatas,     ids=ids ) print(f"Successfully added {collection.count()} documents to ChromaDB\!") |
| :---- |

### **📘 Step 4: ทดสอบการค้นหาแบบ Pure Semantic Search**

ทดสอบรัน Query เดิมจากใบงานก่อนหน้า เพื่อดูผลลัพธ์จาก Vector Search เพียงอย่างเดียว

| import chromadb from chromadb.utils import embedding\_functions \# 1\. Connect to the existing database on disk chroma\_client \= chromadb.PersistentClient(path="./chroma\_db") \# 2\. Retrieve the existing collection created by File 1 \# Note: Use the exact same collection name and embedding\_function as File 1 collection \= chroma\_client.get\_collection(     name="my\_documents"     \# embedding\_function=embedding\_functions.DefaultEmbeddingFunction() \# Optional: Pass if custom EF was used ) \# 3\. Define search function def search\_vector\_db(query\_text, n\_results=2):     results \= collection.query(         query\_texts=\[query\_text\],         n\_results=n\_results     )     return results \# 4\. Run queries queries \= \[     "การขอเบิกเงินค่าเดินทางไปทำงานต่างจังหวัด", \# Q1 (Synonym/Context)     "ขอคู่มือรหัส D-9902"                          \# Q2 (Exact Code Match) \] for q in queries:     print(f"\\n🔍 Query: '{q}'")     res \= search\_vector\_db(q)     for doc, meta, dist in zip(res\['documents'\]\[0\], res\['metadatas'\]\[0\], res\['distances'\]\[0\]):         print(f"  \-\> \[{meta\['doc\_id'\]}\] (Code: {meta\['doc\_code'\]}) Distance: {dist:.4f} | Text: {doc}") |
| :---- |

### **📘 Step 5: แก้ไขปัญหา Exact Match ด้วย Metadata Filtering (สร้างไฟล์ใหม่แยกเพื่อเรียกใช้ ChromaDB)**

ใช้ความสามารถของ Vector Database ในการทำ **Hybrid Search / Metadata Filter** ร่วมกับ Vector Search เมื่อผู้ใช้ต้องการระบุรหัสเอกสารเฉพาะ เช่น D-9902

| \# ค้นหาโดยระบุ Filter ให้ค้นเฉพาะเอกสารที่มี doc\_code \= "D-9902" exact\_search\_result \= collection.query(     query\_texts=\["ขอคู่มือใช้งาน"\],     where={"doc\_code": "D-9902"}, \# Filter Metadata     n\_results=1 ) print("\\n🎯 Search with Metadata Filter (doc\_code \= 'D-9902'):") print("Found Doc:", exact\_search\_result\['documents'\]\[0\]\[0\]) print("Metadata:", exact\_search\_result\['metadatas'\]\[0\]\[0\]) |
| :---- |

### **📝 โจทย์และคำถามท้ายใบงาน (Exercises & Discussion)**

1. **การเปรียบเทียบความคุ้มค่า (DB vs Library):** จากการทดลองใช้งาน ChromaDB เปรียบเทียบกับการคำนวณ Vector ใน Memory จากใบงานเดิม คุณคิดว่า Vector Database มีจุดเด่นอะไรบ้างที่เหมาะกับการทำระบบ RAG จริงในระดับ Production?  
2. **การแก้ไขจุดอ่อนของ Dense Model:** ในโจทย์ Q2 ที่ค้นหารหัสเอกสาร D-9902 การนำ Metadata Filtering ของ ChromaDB มาเสริม ช่วยแก้ปัญหาการค้นหา Exact Term ได้อย่างไร เมื่อเทียบกับการใช้ Pure Dense Search เพียงอย่างเดียว?  
3. **การประยุกต์ใช้เพิ่มเติม:** ให้ทดลองเพิ่มเอกสารใหม่จำนวน 3 เอกสารที่มีประเภท (Category) และรหัสเอกสาร (doc\_code) ต่างกัน แล้วเขียนฟังก์ชันการค้นหาที่ผสมผสานระหว่าง **Semantic Query \+ Metadata Filter (Category)** พร้อมแสดงผลลัพธ์

