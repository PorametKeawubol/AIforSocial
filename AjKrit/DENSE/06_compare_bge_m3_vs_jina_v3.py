# -*- coding: utf-8 -*-

"""
Compare Thai Dense Retrieval Models

Baseline:
    BAAI/bge-m3

New BERT-family model:
    jinaai/jina-embeddings-v3-hf

Compare:
    - Top-1 document
    - Ranking
    - Cosine Distance
"""

from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "chroma_db_model_compare"

BGE_MODEL_NAME = "BAAI/bge-m3"
JINA_MODEL_NAME = "jinaai/jina-embeddings-v3-hf"

BGE_COLLECTION_NAME = "thai_rag_bge_m3"
JINA_COLLECTION_NAME = "thai_rag_jina_v3"

OUTPUT_CSV = BASE_DIR / "bge_m3_vs_jina_v3_distance.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# DATASET
# Same documents as ChromaDB Lab
# ============================================================

documents = [
    "นโยบายการเบิกจ่ายค่าเดินทางและค่าเบี้ยเลี้ยงต่างจังหวัดประจำปี 2024",

    "คู่มือการใช้งานระบบสารสนเทศรหัสเอกสาร D-9902 "
    "สำหรับเจ้าหน้าที่ IT",

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
# TEST QUERIES
# ============================================================

queries = [
    {
        "name": "Q1",
        "type": "Semantic / Context",
        "text": "การขอเบิกเงินค่าเดินทางไปทำงานต่างจังหวัด",
        "ground_truth": "Doc 1"
    },

    {
        "name": "Q2",
        "type": "Exact Code",
        "text": "ขอคู่มือรหัส D-9902",
        "ground_truth": "Doc 3"
    }
]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_numpy(embeddings):

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    norm = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norm = np.maximum(norm, 1e-12)

    return embeddings / norm


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("=" * 100)
print("LOADING EMBEDDING MODELS")
print("=" * 100)

print(f"\nDevice: {DEVICE}")


# ------------------------------------------------------------
# BGE-M3
# ------------------------------------------------------------

print("\nLoading BAAI/bge-m3 ...")

bge_model = SentenceTransformer(
    BGE_MODEL_NAME,
    device=DEVICE
)

print("Loaded:", BGE_MODEL_NAME)


# ------------------------------------------------------------
# JINA EMBEDDINGS V3 - NATIVE HF
# ------------------------------------------------------------

print("\nLoading jinaai/jina-embeddings-v3-hf ...")

jina_tokenizer = AutoTokenizer.from_pretrained(
    JINA_MODEL_NAME
)

jina_model = AutoModel.from_pretrained(
    JINA_MODEL_NAME,
    dtype=torch.float32
)

jina_model = jina_model.to(DEVICE)

print("Base model loaded.")


# ============================================================
# LOAD JINA TASK ADAPTERS
# ============================================================

print("\nLoading Jina retrieval adapters ...")


# Passage adapter
jina_model.load_adapter(
    JINA_MODEL_NAME,
    adapter_name="retrieval_passage",
    adapter_kwargs={
        "subfolder": "retrieval_passage"
    }
)

print("Loaded adapter: retrieval_passage")


# Query adapter
jina_model.load_adapter(
    JINA_MODEL_NAME,
    adapter_name="retrieval_query",
    adapter_kwargs={
        "subfolder": "retrieval_query"
    }
)

print("Loaded adapter: retrieval_query")


jina_model.eval()

print("Loaded:", JINA_MODEL_NAME)


# ============================================================
# JINA ENCODER
# ============================================================

def jina_encode(texts, task):
    """
    task:
        retrieval_passage -> documents
        retrieval_query   -> queries

    Jina v3 uses task-specific LoRA adapters.
    """

    jina_model.set_adapter(task)

    encoded = jina_tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        output = jina_model(
            **encoded
        )

        # Token embeddings
        token_embeddings = output.last_hidden_state

        # Attention mask
        attention_mask = (
            encoded["attention_mask"]
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        # Mean Pooling
        embeddings = torch.sum(
            token_embeddings * attention_mask,
            dim=1
        )

        mask_sum = torch.clamp(
            attention_mask.sum(dim=1),
            min=1e-9
        )

        embeddings = embeddings / mask_sum

        # L2 Normalize
        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1
        )

    return (
        embeddings
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ============================================================
# DOCUMENT EMBEDDINGS
# ============================================================

print()
print("=" * 100)
print("ENCODING DOCUMENTS")
print("=" * 100)


# ------------------------------------------------------------
# BGE DOCUMENTS
# ------------------------------------------------------------

print("\nEncoding documents with BGE-M3 ...")

bge_doc_embeddings = bge_model.encode(
    documents,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=False
)

bge_doc_embeddings = normalize_numpy(
    bge_doc_embeddings
)


# ------------------------------------------------------------
# JINA DOCUMENTS
# Important: retrieval_passage
# ------------------------------------------------------------

print("Encoding documents with Jina v3 (retrieval_passage) ...")

jina_doc_embeddings = jina_encode(
    documents,
    task="retrieval_passage"
)


print()
print(
    "BGE-M3 embedding dimension:",
    bge_doc_embeddings.shape[1]
)

print(
    "Jina v3 embedding dimension:",
    jina_doc_embeddings.shape[1]
)


# ============================================================
# CHROMADB
# ============================================================

print()
print("=" * 100)
print("CREATING CHROMADB COLLECTIONS")
print("=" * 100)

client = chromadb.PersistentClient(
    path=str(DB_PATH)
)


# Delete old collections
for name in [
    BGE_COLLECTION_NAME,
    JINA_COLLECTION_NAME
]:

    try:
        client.delete_collection(
            name=name
        )
    except Exception:
        pass


# ============================================================
# BGE COLLECTION
# ============================================================

bge_collection = client.create_collection(

    name=BGE_COLLECTION_NAME,

    metadata={
        "hnsw:space": "cosine"
    }
)


bge_collection.add(

    ids=ids,

    documents=documents,

    metadatas=metadatas,

    embeddings=bge_doc_embeddings.tolist()
)


# ============================================================
# JINA COLLECTION
# ============================================================

jina_collection = client.create_collection(

    name=JINA_COLLECTION_NAME,

    metadata={
        "hnsw:space": "cosine"
    }
)


jina_collection.add(

    ids=ids,

    documents=documents,

    metadatas=metadatas,

    embeddings=jina_doc_embeddings.tolist()
)


print(
    f"\nBGE-M3 collection: "
    f"{bge_collection.count()} documents"
)

print(
    f"Jina v3 collection: "
    f"{jina_collection.count()} documents"
)


# ============================================================
# SEARCH FUNCTIONS
# ============================================================

def search_bge(query_text):

    embedding = bge_model.encode(

        [query_text],

        normalize_embeddings=True,

        convert_to_numpy=True,

        show_progress_bar=False
    )

    embedding = normalize_numpy(
        embedding
    )

    return bge_collection.query(

        query_embeddings=embedding.tolist(),

        n_results=3,

        include=[
            "metadatas",
            "documents",
            "distances"
        ]
    )


def search_jina(query_text):

    # Important:
    # Query uses retrieval_query adapter
    embedding = jina_encode(

        [query_text],

        task="retrieval_query"
    )

    return jina_collection.query(

        query_embeddings=embedding.tolist(),

        n_results=3,

        include=[
            "metadatas",
            "documents",
            "distances"
        ]
    )


# ============================================================
# RUN COMPARISON
# ============================================================

print()
print()
print("=" * 100)
print("BGE-M3 vs JINA-EMBEDDINGS-V3-HF")
print("=" * 100)

results = []


for query_info in queries:

    query_name = query_info["name"]
    query_type = query_info["type"]
    query_text = query_info["text"]
    ground_truth = query_info["ground_truth"]


    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    bge_result = search_bge(
        query_text
    )

    jina_result = search_jina(
        query_text
    )


    # --------------------------------------------------------
    # TOP-1
    # --------------------------------------------------------

    bge_meta = (
        bge_result["metadatas"][0][0]
    )

    bge_distance = float(
        bge_result["distances"][0][0]
    )


    jina_meta = (
        jina_result["metadatas"][0][0]
    )

    jina_distance = float(
        jina_result["distances"][0][0]
    )


    bge_correct = (
        bge_meta["doc_id"] == ground_truth
    )

    jina_correct = (
        jina_meta["doc_id"] == ground_truth
    )


    # --------------------------------------------------------
    # PRINT QUERY
    # --------------------------------------------------------

    print()
    print("=" * 100)

    print(
        f"{query_name} ({query_type})"
    )

    print(
        f"Query: {query_text}"
    )

    print(
        f"Ground Truth: {ground_truth}"
    )

    print("=" * 100)


    # --------------------------------------------------------
    # TOP-1 TABLE
    # --------------------------------------------------------

    print()
    print("TOP-1 COMPARISON")

    print("-" * 100)

    print(
        f"{'Model':32}"
        f"{'Top-1':12}"
        f"{'Doc Code':15}"
        f"{'Distance':15}"
        f"{'Correct'}"
    )

    print("-" * 100)


    print(
        f"{'BAAI/bge-m3':32}"
        f"{bge_meta['doc_id']:12}"
        f"{bge_meta['doc_code']:15}"
        f"{bge_distance:<15.6f}"
        f"{'YES' if bge_correct else 'NO'}"
    )


    print(
        f"{'jina-embeddings-v3-hf':32}"
        f"{jina_meta['doc_id']:12}"
        f"{jina_meta['doc_code']:15}"
        f"{jina_distance:<15.6f}"
        f"{'YES' if jina_correct else 'NO'}"
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    results.append({

        "Query":
            query_name,

        "Query_Type":
            query_type,

        "Query_Text":
            query_text,

        "Ground_Truth":
            ground_truth,

        "BGE_Top1":
            bge_meta["doc_id"],

        "BGE_Code":
            bge_meta["doc_code"],

        "BGE_Distance":
            bge_distance,

        "BGE_Correct":
            bge_correct,

        "Jina_Top1":
            jina_meta["doc_id"],

        "Jina_Code":
            jina_meta["doc_code"],

        "Jina_Distance":
            jina_distance,

        "Jina_Correct":
            jina_correct
    })


    # --------------------------------------------------------
    # BGE RANKING
    # --------------------------------------------------------

    print()
    print("BGE-M3 Ranking")
    print("-" * 70)


    for rank, (
        metadata,
        distance
    ) in enumerate(

        zip(
            bge_result["metadatas"][0],
            bge_result["distances"][0]
        ),

        start=1
    ):

        print(
            f"Rank {rank} | "
            f"{metadata['doc_id']} | "
            f"{metadata['doc_code']} | "
            f"Distance = {distance:.6f}"
        )


    # --------------------------------------------------------
    # JINA RANKING
    # --------------------------------------------------------

    print()
    print("Jina Embeddings v3 Ranking")
    print("-" * 70)


    for rank, (
        metadata,
        distance
    ) in enumerate(

        zip(
            jina_result["metadatas"][0],
            jina_result["distances"][0]
        ),

        start=1
    ):

        print(
            f"Rank {rank} | "
            f"{metadata['doc_id']} | "
            f"{metadata['doc_code']} | "
            f"Distance = {distance:.6f}"
        )


# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(
    results
)

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 100)
print("FINAL COMPARISON")
print("=" * 100)

print()

print(
    df[
        [
            "Query",
            "Ground_Truth",
            "BGE_Top1",
            "BGE_Distance",
            "Jina_Top1",
            "Jina_Distance"
        ]
    ].to_string(
        index=False
    )
)


print()
print("CSV saved:")
print(OUTPUT_CSV)

print()
print("=" * 100)
print("DONE")
print("=" * 100)