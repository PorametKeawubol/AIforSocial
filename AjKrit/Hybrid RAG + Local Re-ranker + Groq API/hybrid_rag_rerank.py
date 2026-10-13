import os
from typing import List
from sentence_transformers import CrossEncoder

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------------------------
# Step 0: ตรวจสอบ API Key
# --------------------------------------------------------------
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("กรุณาตั้งค่า GROQ_API_KEY ใน Environment Variable ก่อนเริ่มใช้งาน")

# --------------------------------------------------------------
# Step 1: เตรียมข้อมูลเอกสาร (Document Loading & Chunking)
# --------------------------------------------------------------
# สมมติตัวอย่างชุดข้อมูล (ในงานจริงสามารถใช้ PyPDFLoader หรือ DirectoryLoader ได้)
sample_documents = [
    "Groq ใช้ชิป LPU (Language Processing Unit) ช่วยให้การประมวลผลภาษาของ LLM รวดเร็วมากถึง 500-800 tokens ต่อวินาที",
    "การทำ Hybrid RAG เป็นการผสานข้อดีระหว่าง BM25 (ค้นหาด้วย Keyword) และ Vector Search (ค้นหาด้วยความหมาย)",
    "Local Re-ranker โดยใช้ Cross-Encoder ช่วยวัดความสัมพันธ์ระหว่างคำถามกับเอกสารได้อย่างแม่นยำสูงกว่า Bi-Encoder",
    "การใช้ Cross-Encoder คัดกรองเอกสารให้เหลือเพียง 3-5 Chunks ช่วยลดโอกาสเกิดอาการหลอน (Hallucination) ของ LLM",
    "โมเดล llama-3.1-8b-instant บน Groq เหมาะสำหรับการเรียบเรียงประโยคตอบกลับที่ผ่านการ Re-rank มาแล้วอย่างยิ่ง",
    "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย มีสภาพอากาศแบบร้อนชื้น"
]

print("1. กำลังแบ่งเนื้อหาเอกสารเป็น Chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = text_splitter.create_documents(sample_documents)

# --------------------------------------------------------------
# Step 2: สร้าง Local Hybrid Retrievers (Dense + Sparse)
# --------------------------------------------------------------
print("2. กำลังโหลด Local Embedding Model และสร้าง Vector Store...")
# ใช้โมเดล Embedding ฝั่ง Local (รองรับหลายภาษา/ภาษาไทย)
local_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"} # หากมี GPU ให้เปลี่ยนเป็น "cuda"
)

# 2a. Dense Vector Search (ChromaDB)
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=local_embeddings,
    collection_name="local_hybrid_rag"
)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 2b. Sparse Keyword Search (BM25)
sparse_retriever = BM25Retriever.from_documents(docs)
sparse_retriever.k = 10

# 2c. ผสานรวมเป็น Hybrid Retriever (ดึงมาเบื้องต้น 10-20 Chunks)
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5]
)

# --------------------------------------------------------------
# Step 3: นิยาม Local Cross-Encoder Re-ranker Class
# --------------------------------------------------------------
class LocalReranker:
    """Class สำหรับทำ Re-ranking เอกสารบนเครื่อง local ด้วย Cross-Encoder"""
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        print(f"3. กำลังโหลด Local Cross-Encoder Re-ranker ({model_name})...")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Document], top_n: int = 3) -> List[Document]:
        if not documents:
            return []
        
        # จับคู่ Query กับเนื้อหาแต่ละ Document
        pairs = [[query, doc.page_content] for doc in documents]
        
        # คำนวณคะแนนความสัมพันธ์ (Relevance Scores)
        scores = self.model.predict(pairs)
        
        # ผูกคะแนนเข้ากับ Document แล้วเรียงลำดับจากมากไปน้อย
        doc_score_pairs = list(zip(documents, scores))
        sorted_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        
        # คัดเอาเฉพาะ Top-N ที่คะแนนสูงสุด
        reranked_docs = [doc for doc, score in sorted_pairs[:top_n]]
        return reranked_docs

# เรียกใช้งาน Re-ranker
reranker = LocalReranker(model_name="BAAI/bge-reranker-base")

# --------------------------------------------------------------
# Step 4: กำหนดค่า Groq LLM (ใช้โมเดลขนาดเล็ก)
# --------------------------------------------------------------
print("4. กำลังเชื่อมต่อกับ Groq Cloud API...")
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # โมเดลขนาดเล็ก รวดเร็ว และประหยัด Quota
    temperature=0.1,               # ค่าน้อยเพื่อให้อิงตาม Context เป็นหลัก
    max_retries=2
)

# สร้าง Prompt Template
prompt_template = """คุณคือผู้ช่วยตอบคำถาม โปรดตอบคำถามโดยใช้อ้อมูลจากบริบท (Context) ที่กำหนดให้เท่านั้น 
หากในบริบทไม่มีข้อมูล ให้ตอบว่า "ไม่พบข้อมูลในเอกสารที่กำหนด" โดยอย่าพยายามสร้างข้อมูลขึ้นเอง

<context>
{context}
</context>

คำถาม: {question}
คำตอบ:"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# --------------------------------------------------------------
# Step 5: ฟังก์ชันสร้าง RAG Pipeline แบบรวมขั้นตอน
# --------------------------------------------------------------
def run_hybrid_rag_pipeline(query: str, top_n_rerank: int = 3):
    print(f"\n==================================================")
    print(f" User Query: {query}")
    print(f"==================================================")
    
    # 1. Hybrid Retrieval (ดึงมากว้างๆ)
    initial_docs = hybrid_retriever.invoke(query)
    print(f"-> ดึงข้อมูลจาก Hybrid Search ได้: {len(initial_docs)} Chunks")
    
    # 2. Local Re-ranking (คัดเฉพาะเน้นๆ)
    reranked_docs = reranker.rerank(query, initial_docs, top_n=top_n_rerank)
    print(f"-> ผ่าน Re-ranker คัดเหลือเฉพาะ Top {len(reranked_docs)} Chunks ที่ตรงที่สุด")
    
    # รวม Context
    context_text = "\n\n".join([f"- {doc.page_content}" for doc in reranked_docs])
    
    print("\n--- [Context ที่ส่งให้ Groq] ---")
    print(context_text)
    print("--------------------------------\n")
    
    # 3. ส่งให้ Groq สรุปคำตอบ
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context_text, "question": query})
    
    return response

# --------------------------------------------------------------
# Step 6: ทดสอบการรันระบบ
# --------------------------------------------------------------
if __name__ == "__main__":
    test_query = "ทำไมการทำ Re-rank และใช้โมเดลเล็กบน Groq ถึงช่วยลด Hallucination ได้?"
    answer = run_hybrid_rag_pipeline(test_query, top_n_rerank=2)
    
    print("=== คำตอบจาก Groq (llama-3.1-8b-instant) ===")
    print(answer)
    print("===========================================")

