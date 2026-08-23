# -*- coding: utf-8 -*-

from chroma_config import (
    get_client,
    get_model,
    encode_documents,
    COLLECTION_NAME,
    MODEL_NAME
)


# ============================================================
# STEP 2: LOAD DENSE MODEL
# ============================================================

print("=" * 70)
print("STEP 2: LOAD DENSE MODEL")
print("=" * 70)

model = get_model()

print("Loaded Dense Model:")
print(MODEL_NAME)


# ============================================================
# STEP 3: DOCUMENTS + METADATA
# ============================================================

documents = [
    "นโยบายการเบิกจ่ายค่าเดินทางและค่าเบี้ยเลี้ยงต่างจังหวัดประจำปี 2024",

    "คู่มือการใช้งานระบบสารสนเทศรหัสเอกสาร D-9902 สำหรับเจ้าหน้าที่ IT",

    "แนวทางการขออนุญาตลาพักร้อนและการลาป่วยผ่านระบบออนไลน์"
]


metadatas = [
    {
        "doc_id": "Doc 1",
        "category": "HR",
        "doc_code": "D-1001"
    },

    {
        "doc_id": "Doc 3",
        "category": "IT",
        "doc_code": "D-9902"
    },

    {
        "doc_id": "Doc 2",
        "category": "HR",
        "doc_code": "D-1002"
    }
]


ids = [
    "doc_1",
    "doc_3",
    "doc_2"
]


# ============================================================
# CONNECT CHROMADB
# ============================================================

client = get_client()


# ============================================================
# RESET COLLECTION
# ============================================================
#
# ลบ collection เดิมก่อน
# เพื่อให้สามารถรันไฟล์นี้ซ้ำได้โดยไม่เกิดข้อมูลซ้ำ
# ============================================================

try:
    client.delete_collection(
        name=COLLECTION_NAME
    )

    print()
    print(
        f"Deleted old collection: "
        f"{COLLECTION_NAME}"
    )

except Exception:
    pass


# ============================================================
# CREATE COLLECTION
# ============================================================

collection = client.create_collection(
    name=COLLECTION_NAME,

    # ตามใบงาน กำหนด metric เป็น cosine
    metadata={
        "hnsw:space": "cosine"
    }
)


print()
print(
    f"Created collection: "
    f"{COLLECTION_NAME}"
)


# ============================================================
# CREATE DOCUMENT EMBEDDINGS
# ============================================================

print()
print("Encoding documents...")


embeddings = encode_documents(
    model,
    documents
)


print(
    "Embedding dimension:",
    len(embeddings[0])
)


# ============================================================
# ADD TO CHROMADB
# ============================================================

collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)


print()
print(
    f"Successfully added "
    f"{collection.count()} documents "
    f"to ChromaDB!"
)


# ============================================================
# DISPLAY DATA
# ============================================================

print()
print("=" * 70)
print("DOCUMENTS")
print("=" * 70)


for doc_id, doc, metadata in zip(
    ids,
    documents,
    metadatas
):
    print()
    print("ID       :", doc_id)
    print("Doc ID   :", metadata["doc_id"])
    print("Category :", metadata["category"])
    print("Doc Code :", metadata["doc_code"])
    print("Text     :", doc)