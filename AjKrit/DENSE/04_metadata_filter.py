# -*- coding: utf-8 -*-

from chroma_config import (
    get_client,
    get_model,
    encode_query,
    COLLECTION_NAME
)


# ============================================================
# STEP 5: EXACT MATCH USING METADATA FILTER
# ============================================================

print("=" * 70)
print("STEP 5: METADATA FILTER")
print("=" * 70)


# ============================================================
# CONNECT DATABASE
# ============================================================

client = get_client()


collection = client.get_collection(
    name="thai_rag_docs"
)


print(
    "Connected collection:",
    COLLECTION_NAME
)


# ============================================================
# LOAD MODEL
# ============================================================

model = get_model()


# ============================================================
# QUERY
# ============================================================

query_text = "ขอคู่มือใช้งาน"


query_embedding = encode_query(
    model,
    query_text
)


# ============================================================
# SEMANTIC SEARCH + EXACT METADATA FILTER
# ============================================================

results = collection.query(

    query_embeddings=query_embedding,

    where={
        "doc_code": "D-9902"
    },

    n_results=1
)


# ============================================================
# OUTPUT
# ============================================================

print()
print(
    "Query:",
    query_text
)

print(
    "Metadata Filter: "
    "doc_code = D-9902"
)


if len(results["documents"][0]) > 0:

    document = (
        results["documents"][0][0]
    )

    metadata = (
        results["metadatas"][0][0]
    )

    distance = (
        results["distances"][0][0]
    )


    print()
    print("FOUND DOCUMENT")

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

else:

    print(
        "No document found."
    )