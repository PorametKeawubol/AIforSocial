# -*- coding: utf-8 -*-

from chroma_config import (
    get_client,
    get_model,
    encode_query,
    COLLECTION_NAME,
    MODEL_NAME
)


# ============================================================
# STEP 4: PURE SEMANTIC SEARCH
# ============================================================

print("=" * 70)
print("STEP 4: PURE SEMANTIC SEARCH")
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

print(
    "Loaded model:",
    MODEL_NAME
)


# ============================================================
# SEARCH FUNCTION
# ============================================================

def search_vector_db(
    query_text,
    n_results=2
):
    """
    Pure Dense Semantic Search
    """

    query_embedding = encode_query(
        model,
        query_text
    )

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    return results


# ============================================================
# QUERIES
# ============================================================

queries = [
    # Q1 Synonym / Context
    "การขอเบิกเงินค่าเดินทางไปทำงานต่างจังหวัด",

    # Q2 Exact Code
    "ขอคู่มือรหัส D-9902"
]


# ============================================================
# SEARCH
# ============================================================

for q in queries:

    print()
    print("=" * 70)
    print("Query:", q)
    print("=" * 70)

    results = search_vector_db(
        q,
        n_results=2
    )


    for rank, (
        doc,
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
            doc
        )