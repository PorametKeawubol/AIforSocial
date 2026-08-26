# **🧪 RAG-Based QA Lab Sheet (PDF \+ Thai \+ ChatOllama)**

## **🧾 Title:**

**RAG-Based Question Answering from Thai PDF Book Using ChatOllama**  
---

## **📋 Requirements**

* Python 3.9+

* Optional: GPU for faster embedding or LLaMA inference

* PDF document in Thai (e.g., engineering, solar cell, etc.)

---

## **📦 Installation**

Install required packages:  
`pip install sentence-transformers langchain llama-cpp-python chromadb pypdf`

---

## **🧱 Project Structure**

`rag_lab/`  
`├── solarcell-basic-knowledge-SolarHub.pdf  # Example Thai PDF`  
`├── rag_pdf_chatollama.py                   # Your main lab code`

# เริ่มต้นรู้จักกับโซล่าเซลล์

[https://www.solarhub.co.th/solar-information/solar-intro/349-solarcell-basic-knowledge](https://www.solarhub.co.th/solar-information/solar-intro/349-solarcell-basic-knowledge)   
---

## **🧪 Full Lab Code**

**PDF → ChromaDB → Retrieval → Prompt → Ollama → Console Chat** และไม่มี LINE Webhook

| import os import sys from typing import List from PyPDF2 import PdfReader from langchain.schema import Document from langchain.vectorstores import Chroma from langchain.embeddings import HuggingFaceEmbeddings from langchain\_community.chat\_models import ChatOllama \# Configuration PDF\_PATH \= os.environ.get("RAG\_PDF\_PATH", "solarcell-basic-knowledge-SolarHub.pdf") CHROMA\_DIR \= os.environ.get("CHROMA\_DIR", "./chroma\_db") EMBED\_MODEL\_NAME \= os.environ.get("EMBED\_MODEL\_NAME", "paraphrase-multilingual-MiniLM-L12-v2") RETRIEVAL\_K \= int(os.environ.get("RETRIEVAL\_K", "3")) OLLAMA\_MODEL \= os.environ.get("OLLAMA\_MODEL", "qwen2.5:3b") \# 1\. Extract text from PDF def extract\_lines\_from\_pdf(path: str) \-\> List\[str\]:     if not os.path.isfile(path):         print(f"\[ERROR\] PDF not found: {path}")         sys.exit(1)     reader \= PdfReader(path)     lines \= \[\]     for page in reader.pages:         try:             text \= page.extract\_text()         except Exception:             text \= None         if not text:             continue         for line in text.split("\\n"):             line \= line.strip()             if line:                 lines.append(line)     return lines \# 2\. Chunk text def chunk\_lines(lines: List\[str\], chunk\_size: int \= 5, overlap: int \= 2\) \-\> List\[str\]:     chunks \= \[\]     if chunk\_size \<= 0:         return chunks     step \= max(1, chunk\_size \- overlap)     for i in range(0, len(lines), step):         chunk \= "\\n".join(lines\[i:i \+ chunk\_size\])         if chunk.strip():             chunks.append(chunk)     return chunks  \# 3\. Build or load ChromaDB def build\_or\_load\_vectorstore(pdf\_path: str, persist\_dir: str) \-\> Chroma:     os.makedirs(persist\_dir, exist\_ok=True)     print(f"\[EMBEDDING\] Loading model: {EMBED\_MODEL\_NAME}")     embedding \= HuggingFaceEmbeddings(model\_name=EMBED\_MODEL\_NAME)     if os.listdir(persist\_dir):         try:             print(f"\[RAG\] Loading existing ChromaDB: {persist\_dir}")             return Chroma(                 persist\_directory=persist\_dir,                 embedding\_function=embedding             )         except Exception as e:             print(f"\[RAG\] Failed to load ChromaDB: {e}")             print("\[RAG\] Rebuilding ChromaDB...")     print(f"\[RAG\] Extracting PDF: {pdf\_path}")     lines \= extract\_lines\_from\_pdf(pdf\_path)     print(f"\[RAG\] Extracted lines: {len(lines)}")     print("\[RAG\] Creating chunks...")     chunks \= chunk\_lines(lines, chunk\_size=5, overlap=2)     print(f"\[RAG\] Created chunks: {len(chunks)}")     documents \= \[         Document(             page\_content=chunk,             metadata={                 "source": os.path.basename(pdf\_path),                 "chunk\_id": str(i)             }         )         for i, chunk in enumerate(chunks)     \]     print("\[RAG\] Building ChromaDB...")     vectorstore \= Chroma.from\_documents(         documents=documents,         embedding=embedding,         persist\_directory=persist\_dir     )     vectorstore.persist()     print("\[RAG\] ChromaDB created successfully.")     return vectorstore \# 4\. Ollama LLM def build\_chat\_llm():     print(f"\[LLM\] Using Ollama model: {OLLAMA\_MODEL}")     try:         return ChatOllama(             model=OLLAMA\_MODEL,             temperature=0.2         )     except Exception as e:         print(f"\[LLM ERROR\] {e}")         raise \# 5\. Prompt def build\_prompt(context: str, question: str) \-\> str:     return f""" คุณคือผู้ช่วยด้านวิศวกรรมพลังงานแสงอาทิตย์ (Solar Energy Engineer) หน้าที่ของคุณคือการตอบคำถามโดยอ้างอิงจากข้อมูลใน Context ที่กำหนดให้เท่านั้น CONTEXT: {context} QUESTION: {question} INSTRUCTIONS: 1\. ตอบเป็นภาษาไทย 2\. ใช้ข้อมูลจาก Context เป็นหลัก 3\. ห้ามสร้างข้อมูลที่ไม่มีอยู่ใน Context 4\. หาก Context ไม่มีข้อมูลเพียงพอ ให้ตอบว่า "ไม่พบข้อมูลที่เพียงพอในเอกสารสำหรับตอบคำถามนี้" 5\. อธิบายให้เข้าใจง่าย เหมาะสำหรับผู้เริ่มต้น 6\. หากมีคำศัพท์ทางวิศวกรรม ให้คงคำศัพท์ภาษาอังกฤษไว้ในวงเล็บ 7\. ตอบให้กระชับ แต่มีรายละเอียดที่จำเป็น 8\. หากเหมาะสม สามารถใช้ emoji เช่น ☀️ 🔋 🔌 ⚡ 9\. ห้ามกล่าวถึงกระบวนการ RAG หรือระบบ Retrieval ภายใน 10\. ห้ามอ้างอิงความรู้จากภายนอก Context ANSWER: """.strip()  \# 6\. RAG Answer def make\_rag\_answer(vectorstore: Chroma, chat\_llm: ChatOllama, question: str, k: int \= 3\) \-\> str:     print(f"\\n\[RAG\] Question: {question}")     retriever \= vectorstore.as\_retriever(search\_kwargs={"k": k})     try:         docs \= retriever.invoke(question)     except Exception as e:         return f"\[Retrieval error\] {e}"     if not docs:         context \= "\[No relevant document found\]"     else:         context \= "\\n\\n---\\n\\n".join(doc.page\_content for doc in docs)     print("\\n\[RAG\] Retrieved Context")     print("=" \* 60\)     for i, doc in enumerate(docs, start=1):         print(f"\\n--- Document {i} \---")         print(doc.page\_content)     print("=" \* 60\)     prompt \= build\_prompt(context, question)     print("\\n\[LLM\] Generating answer...")     try:         response \= chat\_llm.invoke(prompt)         answer \= getattr(response, "content", None) or str(response)         answer \= answer.strip()         return answer if answer else "\[ERROR\] Empty response from LLM."     except Exception as e:         return f"\[LLM error\] {e}" \# 7\. Chat loop def chat\_loop(vectorstore: Chroma, chat\_llm: ChatOllama):     print("=" \* 60\)     print("☀️ Solar Cell RAG Chatbot")     print("=" \* 60\)     print(f"PDF       : {PDF\_PATH}")     print(f"Embedding : {EMBED\_MODEL\_NAME}")     print(f"LLM       : {OLLAMA\_MODEL}")     print(f"Top-K     : {RETRIEVAL\_K}")     print("Type 'exit' or 'quit' to stop.")     print("=" \* 60\)     while True:         try:             question \= input("\\n👤 You: ").strip()         except KeyboardInterrupt:             print("\\nBye\! 👋")             break         if not question:             continue         if question.lower() in {"exit", "quit"}:             print("\\nBye\! 👋")             break         answer \= make\_rag\_answer(             vectorstore=vectorstore,             chat\_llm=chat\_llm,             question=question,             k=RETRIEVAL\_K         )         print("\\n🤖 Assistant:")         print(answer) \# 8\. Main if \_\_name\_\_ \== "\_\_main\_\_":     print("\[BOOT\] Loading Vector Store...")     vectorstore \= build\_or\_load\_vectorstore(         PDF\_PATH,         CHROMA\_DIR     )     print("\[BOOT\] Initializing Ollama...")     chat\_llm \= build\_chat\_llm()     print("\[BOOT\] Starting RAG Chat...")     chat\_loop(         vectorstore,         chat\_llm     ) |
| :---- |

จุดสำคัญคือโค้ดนี้ **ไม่มี Flask และ LINE SDK แล้ว** ดังนั้นรันตรง ๆ ได้ด้วย:

python app.py

แล้วถามผ่าน terminal เช่น:

👤 You: Solar Cell คืออะไร?

🤖 Assistant:  
Solar Cell คืออุปกรณ์ที่เปลี่ยนพลังงานแสงอาทิตย์เป็นพลังงานไฟฟ้า...

