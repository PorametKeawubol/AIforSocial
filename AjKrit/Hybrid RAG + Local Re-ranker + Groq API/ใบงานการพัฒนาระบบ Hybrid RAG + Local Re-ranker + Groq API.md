# **ใบงานการพัฒนาระบบ Hybrid RAG \+ Local Re-ranker \+ Groq API**

## **1\. วัตถุประสงค์การเรียนรู้ (Learning Objectives)**

1. **ด้านสถาปัตยกรรมระบบ (System Architecture):** สามารถออกแบบและสร้างสถาปัตยกรรม Retrieval-Augmented Generation (RAG) ที่ผสานการทำงานระหว่าง Local Processing และ Cloud LLM API ได้อย่างถูกต้อง  
2. **ด้านอัลกอริทึมการค้นคืน (Retrieval Algorithms):** เข้าใจและสามารถเปรียบเทียบประสิทธิภาพระหว่าง Sparse Retrieval (BM25) และ Dense Retrieval (Vector Embedding) ในรูปแบบ Hybrid Search ได้  
3. **ด้านการเพิ่มประสิทธิภาพด้วย Re-ranking (Optimization):** เข้าใจหลักการทำงานของ Cross-Encoder Model ในการคัดกรองข้อมูลเพื่อลดมิติข้อมูล Context ก่อนส่งให้ Large Language Model (LLM)  
4. **ด้านการวัดผลและวิเคราะห์ (Performance Evaluation):** สามารถวัดผลประสิทธิภาพเชิงวิศวกรรม ได้แก่ เวลาแฝง (Latency), ปริมาณการใช้ทรัพยากร Token (Token Consumption) และความแม่นยำของคำตอบ

## **2\. ทฤษฎีและหลักการที่เกี่ยวข้อง (Theoretical Background)**

### **2.1 Hybrid Search & Reciprocal Rank Fusion (RRF)**

การค้นคืนข้อมูลแบบดั้งเดิมแบ่งออกเป็น 2 ประเภทหลัก:

* **Sparse Retrieval (BM25):** คำนวณความถี่ของคำ (Term Frequency) และความถี่ผกผันในเอกสาร (Inverse Document Frequency) เหมาะสำหรับการค้นหาคำเฉพาะ (Keywords)  
* **Dense Retrieval (Vector Embeddings):** แปลงข้อความเป็น Vector ในความหนาแน่นสูงเพื่อหาความเหมือนทางเวกเตอร์ (Cosine Similarity) เหมาะสำหรับการค้นหาเชิงความหมาย (Semantic Search)

การทำ Hybrid Search นำผลลัพธ์จากทั้งสองระบบมารวมกัน โดยใช้เทคนิค Reciprocal Rank Fusion (RRF) ในการให้คะแนนใหม่:

![][image1]

เมื่อ M คือชุดของระบบการค้นคืน (Retrievers), rm(d) คือลำดับของเอกสาร d ในระบบ m, และ k คือค่าคงที่ปรับราบ (Smoothing Constant โดยทั่วไปกำหนดเป็น 60\) 

### **2.2 Bi-Encoder vs. Cross-Encoder Re-ranking**

* **Bi-Encoder (Vector Search):** คำนวณ เวกเตอร์ของ Query และ Document แยกกัน ทำให้สามารถทำ Pre-indexing ได้รวดเร็ว ($O(1)$ ต่อ Vector Lookup) แต่เสียความสัมพันธ์เชิงลึกระหว่างคำใน Query และ Document  
* **Cross-Encoder (Re-ranker):** ป้อน Query และ Document เข้าไปใน Transformer Model พร้อมกันเพื่อคำนวณ Attention Cross-Matrix ทำให้มีความแม่นยำสูงมาก แต่มีความซับซ้อนทางการคำนวณสูง ($O(N)$ เมื่อ $N$ คือจำนวน Candidates) จึงนิยมนำมาใช้คัดกรองเฉพาะ Top-K เอกสารในขั้นสุดท้ายเท่านั้น

## **แผนผังการทำงานของระบบ (Architecture)**

 \[User Query\]   
      │  
      ▼  
 ┌──────────────────────────────────────────────────────────────┐  
 │ 1\. Local Hybrid Search                                       │  
 │    ├── Sparse Search (BM25): ดึงด้วย Keyword                 │  
 │    └── Dense Search (Vector DB): ดึงด้วย ความหมาย (Semantic)  │  
 └──────────────────────────────┬───────────────────────────────┘  
                                │ (ดึงมา 10–20 Chunks)  
                                ▼  
 ┌──────────────────────────────────────────────────────────────┐  
 │ 2\. Local Cross-Encoder Re-ranker                            │  
 │    └── เปรียบเทียบ Query กับทุก Chunk ฝั่ง Local                │  
 │    └── เรียงลำดับคะแนนความเกี่ยวข้องใหม่                          │  
 └──────────────────────────────┬───────────────────────────────┘  
                                │ (คัดเหลือเพียง Top 3–5 Chunks)  
                                ▼  
 ┌──────────────────────────────────────────────────────────────┐  
 │ 3\. Groq Cloud API Generation                                 │  
 │    └── Model: llama-3.1-8b-instant                           │  
 │    └── อ่านเฉพาะ Top Chunks สั้นๆ แล้วสร้างประโยคตอบกลับ       │  
 └──────────────────────────────────────────────────────────────┘

## **2\. การเตรียม Environment และติดตั้ง Dependencies**

ติดตั้ง Library ที่จำเป็นผ่าน Terminal/Command Prompt:

Bash  
pip install langchain langchain-community langchain-groq langchain-huggingface langchain-chroma rank\_bm25 sentence-transformers chromadb

ตั้งค่า API Key ของ Groq:

* **Linux / macOS:**  
  Bash  
  export GROQ\_API\_KEY="gsk\_your\_actual\_groq\_api\_key\_here"

* **Windows (Command Prompt):**  
  DOS  
  set GROQ\_API\_KEY=gsk\_your\_actual\_groq\_api\_key\_here

* **Windows (PowerShell):**  
  PowerShell  
  $env:GROQ\_API\_KEY\="gsk\_your\_actual\_groq\_api\_key\_here"

## **3\. ซอร์สโค้ดฉบับสมบูรณ์ (Full Executable Python Code)**

คุณสามารถก๊อปปี้โค้ดด้านล่างนี้ไปบันทึกเป็นไฟล์ hybrid\_rag\_rerank.py แล้วรันได้ทันที:

| import osfrom typing import Listfrom sentence\_transformers import CrossEncoderfrom langchain\_core.documents import Documentfrom langchain\_text\_splitters import RecursiveCharacterTextSplitterfrom langchain\_huggingface import HuggingFaceEmbeddingsfrom langchain\_chroma import Chromafrom langchain\_community.retrievers import BM25Retrieverfrom langchain.retrievers import EnsembleRetrieverfrom langchain\_groq import ChatGroqfrom langchain\_core.prompts import ChatPromptTemplatefrom langchain\_core.output\_parsers import StrOutputParser\# \--------------------------------------------------------------\# Step 0: ตรวจสอบ API Key\# \--------------------------------------------------------------groq\_api\_key \= os.getenv("GROQ\_API\_KEY")if not groq\_api\_key:    raise ValueError("กรุณาตั้งค่า GROQ\_API\_KEY ใน Environment Variable ก่อนเริ่มใช้งาน")\# \--------------------------------------------------------------\# Step 1: เตรียมข้อมูลเอกสาร (Document Loading & Chunking)\# \--------------------------------------------------------------\# สมมติตัวอย่างชุดข้อมูล (ในงานจริงสามารถใช้ PyPDFLoader หรือ DirectoryLoader ได้)sample\_documents \= \[    "Groq ใช้ชิป LPU (Language Processing Unit) ช่วยให้การประมวลผลภาษาของ LLM รวดเร็วมากถึง 500-800 tokens ต่อวินาที",    "การทำ Hybrid RAG เป็นการผสานข้อดีระหว่าง BM25 (ค้นหาด้วย Keyword) และ Vector Search (ค้นหาด้วยความหมาย)",    "Local Re-ranker โดยใช้ Cross-Encoder ช่วยวัดความสัมพันธ์ระหว่างคำถามกับเอกสารได้อย่างแม่นยำสูงกว่า Bi-Encoder",    "การใช้ Cross-Encoder คัดกรองเอกสารให้เหลือเพียง 3-5 Chunks ช่วยลดโอกาสเกิดอาการหลอน (Hallucination) ของ LLM",    "โมเดล llama-3.1-8b-instant บน Groq เหมาะสำหรับการเรียบเรียงประโยคตอบกลับที่ผ่านการ Re-rank มาแล้วอย่างยิ่ง",    "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย มีสภาพอากาศแบบร้อนชื้น"\]print("1. กำลังแบ่งเนื้อหาเอกสารเป็น Chunks...")text\_splitter \= RecursiveCharacterTextSplitter(chunk\_size=300, chunk\_overlap=50)docs \= text\_splitter.create\_documents(sample\_documents)\# \--------------------------------------------------------------\# Step 2: สร้าง Local Hybrid Retrievers (Dense \+ Sparse)\# \--------------------------------------------------------------print("2. กำลังโหลด Local Embedding Model และสร้าง Vector Store...")\# ใช้โมเดล Embedding ฝั่ง Local (รองรับหลายภาษา/ภาษาไทย)local\_embeddings \= HuggingFaceEmbeddings(    model\_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",    model\_kwargs={"device": "cpu"} \# หากมี GPU ให้เปลี่ยนเป็น "cuda")\# 2a. Dense Vector Search (ChromaDB)vectorstore \= Chroma.from\_documents(    documents=docs,    embedding=local\_embeddings,    collection\_name="local\_hybrid\_rag")dense\_retriever \= vectorstore.as\_retriever(search\_kwargs={"k": 10})\# 2b. Sparse Keyword Search (BM25)sparse\_retriever \= BM25Retriever.from\_documents(docs)sparse\_retriever.k \= 10\# 2c. ผสานรวมเป็น Hybrid Retriever (ดึงมาเบื้องต้น 10-20 Chunks)hybrid\_retriever \= EnsembleRetriever(    retrievers=\[dense\_retriever, sparse\_retriever\],    weights=\[0.5, 0.5\])\# \--------------------------------------------------------------\# Step 3: นิยาม Local Cross-Encoder Re-ranker Class\# \--------------------------------------------------------------class LocalReranker:    """Class สำหรับทำ Re-ranking เอกสารบนเครื่อง local ด้วย Cross-Encoder"""    def \_\_init\_\_(self, model\_name: str \= "BAAI/bge-reranker-base"):        print(f"3. กำลังโหลด Local Cross-Encoder Re-ranker ({model\_name})...")        self.model \= CrossEncoder(model\_name)    def rerank(self, query: str, documents: List\[Document\], top\_n: int \= 3\) \-\> List\[Document\]:        if not documents:            return \[\]                \# จับคู่ Query กับเนื้อหาแต่ละ Document        pairs \= \[\[query, doc.page\_content\] for doc in documents\]                \# คำนวณคะแนนความสัมพันธ์ (Relevance Scores)        scores \= self.model.predict(pairs)                \# ผูกคะแนนเข้ากับ Document แล้วเรียงลำดับจากมากไปน้อย        doc\_score\_pairs \= list(zip(documents, scores))        sorted\_pairs \= sorted(doc\_score\_pairs, key=lambda x: x\[1\], reverse=True)                \# คัดเอาเฉพาะ Top-N ที่คะแนนสูงสุด        reranked\_docs \= \[doc for doc, score in sorted\_pairs\[:top\_n\]\]        return reranked\_docs\# เรียกใช้งาน Re-rankerreranker \= LocalReranker(model\_name="BAAI/bge-reranker-base")\# \--------------------------------------------------------------\# Step 4: กำหนดค่า Groq LLM (ใช้โมเดลขนาดเล็ก)\# \--------------------------------------------------------------print("4. กำลังเชื่อมต่อกับ Groq Cloud API...")llm \= ChatGroq(    model="llama-3.1-8b-instant",  \# โมเดลขนาดเล็ก รวดเร็ว และประหยัด Quota    temperature=0.1,               \# ค่าน้อยเพื่อให้อิงตาม Context เป็นหลัก    max\_retries=2)\# สร้าง Prompt Templateprompt\_template \= """คุณคือผู้ช่วยตอบคำถาม โปรดตอบคำถามโดยใช้อ้อมูลจากบริบท (Context) ที่กำหนดให้เท่านั้น หากในบริบทไม่มีข้อมูล ให้ตอบว่า "ไม่พบข้อมูลในเอกสารที่กำหนด" โดยอย่าพยายามสร้างข้อมูลขึ้นเอง\<context\>{context}\</context\>คำถาม: {question}คำตอบ:"""prompt \= ChatPromptTemplate.from\_template(prompt\_template)\# \--------------------------------------------------------------\# Step 5: ฟังก์ชันสร้าง RAG Pipeline แบบรวมขั้นตอน\# \--------------------------------------------------------------def run\_hybrid\_rag\_pipeline(query: str, top\_n\_rerank: int \= 3):    print(f"\\n==================================================")    print(f" User Query: {query}")    print(f"==================================================")        \# 1\. Hybrid Retrieval (ดึงมากว้างๆ)    initial\_docs \= hybrid\_retriever.invoke(query)    print(f"-\> ดึงข้อมูลจาก Hybrid Search ได้: {len(initial\_docs)} Chunks")        \# 2\. Local Re-ranking (คัดเฉพาะเน้นๆ)    reranked\_docs \= reranker.rerank(query, initial\_docs, top\_n=top\_n\_rerank)    print(f"-\> ผ่าน Re-ranker คัดเหลือเฉพาะ Top {len(reranked\_docs)} Chunks ที่ตรงที่สุด")        \# รวม Context    context\_text \= "\\n\\n".join(\[f"- {doc.page\_content}" for doc in reranked\_docs\])        print("\\n--- \[Context ที่ส่งให้ Groq\] \---")    print(context\_text)    print("--------------------------------\\n")        \# 3\. ส่งให้ Groq สรุปคำตอบ    chain \= prompt | llm | StrOutputParser()    response \= chain.invoke({"context": context\_text, "question": query})        return response\# \--------------------------------------------------------------\# Step 6: ทดสอบการรันระบบ\# \--------------------------------------------------------------if \_\_name\_\_ \== "\_\_main\_\_":    test\_query \= "ทำไมการทำ Re-rank และใช้โมเดลเล็กบน Groq ถึงช่วยลด Hallucination ได้?"    answer \= run\_hybrid\_rag\_pipeline(test\_query, top\_n\_rerank=2)        print("=== คำตอบจาก Groq (llama-3.1-8b-instant) \===")    print(answer)    print("===========================================") |
| :---- |

## **4\. ตารางเปรียบเทียบประโยชน์ของการปรับ Architecture**

| หัวข้อ | RAG แบบทั่วไป (ไม่มี Re-rank) | Hybrid RAG \+ Local Re-rank \+ Groq (ตามใบงานนี้) |
| :---- | :---- | :---- |
| **ความแม่นยำของ Context** |  |  |
| **ความเร็วตอบกลับ (Latency)** |  |  |
| **โอกาสเกิด Hallucination** |  |  |
| **ค่าใช้จ่าย / Quota** |  |  |

> 

[image1]: <data:image/png;base64,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>