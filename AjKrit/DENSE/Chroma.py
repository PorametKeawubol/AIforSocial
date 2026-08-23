# -*- coding: utf-8 -*-

import chromadb
from sentence_transformers import SentenceTransformer

# ============================================================
# 1. Persistent ChromaDB
# ============================================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

print("ChromaDB initialized")


# ============================================================
# 2. Dense Model ที่เลือกจากงานก่อน
# ============================================================

MODEL_NAME = "intfloat/multilingual-e5-small"

model = SentenceTransformer(MODEL_NAME)

print(f"Loaded model: {MODEL_NAME}")


# ============================================================
# 3. Documents + Metadata
# ============================================================

documents = [
    "นโยบายการเบิกจ่ายค่าเดินทางและค่าเบี้ยเลี้ยงต่างจังหวัดประจำปี 2024",

    "คู่มือการใช้งานระบบสารสนเทศรหัสเอกสาร D-9902 "
    "สำหรับเจ้าหน้าที่ IT",

    "แนวทางการขออนุญาตลาพักร้อนและการลาป่วยผ่านระบบออนไลน์",
]

metadatas = [
    {
        "doc_id": "Doc 1",
        "category": "HR",
        "doc_code": "D-1001",
    },
    {
        "doc_id": "Doc 3",
        "category": "IT",
        "doc_code": "D-9902",
    },
    {
        "doc_id": "Doc 2",
        "category": "HR",
        "doc_code": "D-1002",
    },
]

ids = [
    "doc_1",
    "doc_3",
    "doc_2",
]


# ============================================================
# E5 DOCUMENT EMBEDDINGS
# ============================================================

doc_embeddings = model.encode(
    ["passage: " + doc for doc in documents],
    normalize_embeddings=True,
).tolist()


# ============================================================
# CREATE COLLECTION
# ============================================================

# ลบ collection เก่าถ้ามี เพื่อให้ทดลองซ้ำง่าย
try:
    client.delete_collection("thai_rag_docs")
except Exception:
    pass


collection = client.create_collection(
    name="thai_rag_docs",
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)


# ใส่ทั้ง document, metadata และ embedding ที่คำนวณเอง
collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=doc_embeddings,
)

print(
    f"Added {collection.count()} documents"
)


# ============================================================
# 4. PURE SEMANTIC SEARCH
# ============================================================

def semantic_search(query_text, n_results=2):

    # E5 QUERY prefix
    query_embedding = model.encode(
        ["query: " + query_text],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )

    return results


queries = [
    "การขอเบิกเงินค่าเดินทางไปทำงานต่างจังหวัด",
    "ขอคู่มือรหัส D-9902",
]


for query in queries:

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    results = semantic_search(
        query,
        n_results=2,
    )

    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):

        print(
            f"[{meta['doc_id']}] "
            f"Code={meta['doc_code']} | "
            f"Category={meta['category']} | "
            f"Distance={distance:.4f}"
        )

        print(doc)


# ============================================================
# 5. EXACT CODE ด้วย METADATA FILTER
# ============================================================

query = "ขอคู่มือใช้งาน"

query_embedding = model.encode(
    ["query: " + query],
    normalize_embeddings=True,
).tolist()


exact_result = collection.query(
    query_embeddings=query_embedding,

    where={
        "doc_code": "D-9902"
    },

    n_results=1,
)


print("\n" + "=" * 80)
print("METADATA FILTER: D-9902")
print("=" * 80)

print(
    "Document:",
    exact_result["documents"][0][0]
)

print(
    "Metadata:",
    exact_result["metadatas"][0][0]
)

print(
    "Distance:",
    exact_result["distances"][0][0]
)