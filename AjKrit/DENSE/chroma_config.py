# -*- coding: utf-8 -*-

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Database จะถูกเก็บใน:
# DENCE/chroma_db/
DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "thai_rag_docs"

MODEL_NAME = "intfloat/multilingual-e5-small"


# ============================================================
# CHROMADB
# ============================================================

def get_client():
    """
    เชื่อมต่อ ChromaDB แบบ Persistent Storage
    """

    return chromadb.PersistentClient(
        path=str(DB_PATH)
    )


# ============================================================
# EMBEDDING MODEL
# ============================================================

def get_model():
    """
    โหลด Dense Model ที่คัดเลือกจากงานก่อนหน้า
    """

    return SentenceTransformer(
        MODEL_NAME
    )


# ============================================================
# E5 EMBEDDING HELPERS
# ============================================================

def encode_documents(model, documents):
    """
    E5 ใช้ prefix 'passage:' สำหรับ document
    """

    texts = [
        "passage: " + doc
        for doc in documents
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embeddings.tolist()


def encode_query(model, query):
    """
    E5 ใช้ prefix 'query:' สำหรับ query
    """

    embeddings = model.encode(
        ["query: " + query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embeddings.tolist()