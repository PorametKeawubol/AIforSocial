# -*- coding: utf-8 -*-

from chroma_config import (
    get_client,
    get_model,
    encode_documents,
    encode_query,
    COLLECTION_NAME
)


# ============================================================
# EXERCISE 3
# Semantic Search + Category Metadata Filter
# ============================================================

print("=" * 70)
print("EXERCISE 3: SEMANTIC + CATEGORY FILTER")
print("=" * 70)


# ============================================================
# CONNECT DATABASE
# ============================================================

client = get_client()


collection = client.get_collection(
    name="thai_rag_docs"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = get_model()


# ============================================================
# ADD 3 NEW DOCUMENTS
# ============================================================

new_documents = [

    "คู่มือการตั้งค่า VPN สำหรับพนักงานที่ต้องการเชื่อมต่อระบบบริษัทจากที่บ้าน",

    "ระเบียบการเบิกค่ารักษาพยาบาลและสวัสดิการด้านสุขภาพสำหรับพนักงาน",

    "คู่มือการสำรองข้อมูลและการกู้คืนระบบฐานข้อมูลขององค์กร"
]


new_metadatas = [

    {
        "doc_id": "Doc 4",
        "category": "IT",
        "doc_code": "D-2001"
    },

    {
        "doc_id": "Doc 5",
        "category": "HR",
        "doc_code": "D-2002"
    },

    {
        "doc_id": "Doc 6",
        "category": "IT",
        "doc_code": "D-2003"
    }
]


new_ids = [
    "doc_4",
    "doc_5",
    "doc_6"
]


# ============================================================
# EMBEDDINGS
# ============================================================

new_embeddings = encode_documents(
    model,
    new_documents
)


# ============================================================
# UPSERT
# ============================================================
#
# ใช้ upsert แทน add
# เพื่อให้รันไฟล์นี้ซ้ำได้
#
# ถ้า id เดิมมีอยู่แล้ว จะ update
# ถ้ายังไม่มี จะ insert
# ============================================================

collection.upsert(

    ids=new_ids,

    documents=new_documents,

    metadatas=new_metadatas,

    embeddings=new_embeddings
)


print()
print(
    f"Collection now contains "
    f"{collection.count()} documents."
)


# ============================================================
# SEARCH FUNCTION
# ============================================================

def search_by_category(
    query_text,
    category,
    n_results=3
):

    query_embedding = encode_query(
        model,
        query_text
    )


    results = collection.query(

        query_embeddings=query_embedding,

        where={
            "category": category
        },

        n_results=n_results
    )


    return results


# ============================================================
# TEST QUERY
# ============================================================

query_text = (
    "วิธีเชื่อมต่อระบบบริษัทจากที่บ้าน"
)

category_filter = "IT"


results = search_by_category(

    query_text,

    category_filter,

    n_results=3
)


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 70)

print(
    "Semantic Query:",
    query_text
)

print(
    "Category Filter:",
    category_filter
)

print("=" * 70)


for rank, (
    document,
    metadata,
    distance
) in enumerate(

    zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ),

    start=1
):

    print()
    print(f"Rank {rank}")

    print(
        "Doc ID   :",
        metadata["doc_id"]
    )

    print(
        "Doc Code :",
        metadata["doc_code"]
    )

    print(
        "Category :",
        metadata["category"]
    )

    print(
        f"Distance : "
        f"{distance:.4f}"
    )

    print(
        "Text     :",
        document
    )