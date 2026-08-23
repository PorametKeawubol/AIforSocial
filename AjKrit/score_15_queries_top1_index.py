# -*- coding: utf-8 -*-
"""
หา "เอกสารที่ใกล้ที่สุดอันดับ 1" เท่านั้น สำหรับแต่ละ query

Output columns:
BM25_score, BM25_index,
DENSE_score, DENSE_index,
RRF_score, RRF_index

หมายเหตุ:
- *_index = ค่า __index_level_0__ ของเอกสารที่ได้อันดับ 1
- ไม่ใช่ rank ของ Ground Truth
- *_score = score ของเอกสารอันดับ 1 นั้น
"""

import os

# ไม่ใช้ TensorFlow ในงานนี้
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pythainlp.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

def resolve_file(preferred_name: str, pattern: str) -> Path:
    preferred = BASE_DIR / preferred_name
    if preferred.exists():
        return preferred

    matches = sorted(BASE_DIR.glob(pattern))
    if matches:
        print(f"ใช้ไฟล์: {matches[0].name}")
        return matches[0]

    return preferred


JSON_FILE = resolve_file(
    "thai_qa_utf8.json",
    "thai_qa_utf8*.json",
)

CSV_FILE = resolve_file(
    "thai_qa_paraphrase_15.csv",
    "thai_qa_paraphrase_15*.csv",
)

MODEL_NAME = "intfloat/multilingual-e5-small"
RRF_K = 60

EMBED_CACHE = BASE_DIR / "thai_instruction_e5_embeddings.npy"
OUTPUT_FILE = BASE_DIR / "thai_qa_paraphrase_15_top1.csv"


# ============================================================
# CHECK FILES
# ============================================================

if not JSON_FILE.exists():
    raise FileNotFoundError(
        f"ไม่พบไฟล์ thai_qa_utf8*.json ใน {BASE_DIR}"
    )

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"ไม่พบไฟล์ thai_qa_paraphrase_15*.csv ใน {BASE_DIR}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("[1/6] Loading data...")

with JSON_FILE.open("r", encoding="utf-8") as f:
    corpus = json.load(f)

query_df = pd.read_csv(CSV_FILE)

documents = [
    str(row.get("instruction", "") or "")
    for row in corpus
]

n_docs = len(documents)

print(f"  Corpus : {n_docs:,}")
print(f"  Queries: {len(query_df)}")


# ============================================================
# BM25
# ============================================================

print("[2/6] Thai tokenization...")

tokenized_docs = [
    word_tokenize(doc, engine="newmm")
    for doc in documents
]

print("[3/6] Building BM25...")
bm25 = BM25Okapi(tokenized_docs)


# ============================================================
# DENSE EMBEDDINGS
# ============================================================

print(f"[4/6] Loading Dense model: {MODEL_NAME}")
dense_model = SentenceTransformer(MODEL_NAME)

doc_embeddings = None

if EMBED_CACHE.exists():
    try:
        cached = np.load(EMBED_CACHE)

        if cached.ndim == 2 and cached.shape[0] == n_docs:
            doc_embeddings = cached
            print(
                f"  Loaded cache: {EMBED_CACHE.name} "
                f"{doc_embeddings.shape}"
            )
        else:
            print("  Embedding cache shape ไม่ตรงกับ corpus -> สร้างใหม่")
    except Exception as e:
        print(f"  อ่าน embedding cache ไม่สำเร็จ: {e}")

if doc_embeddings is None:
    print("[5/6] Encoding corpus embeddings...")

    # E5 format ตามใบงาน
    passages = ["passage: " + doc for doc in documents]

    doc_embeddings = dense_model.encode(
        passages,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    np.save(EMBED_CACHE, doc_embeddings)
    print(f"  Saved cache: {EMBED_CACHE.name}")
else:
    print("[5/6] Skip encoding (using cache)")


# ============================================================
# TOP-1 SEARCH
# ============================================================

print("[6/6] Searching TOP-1 nearest document...")

results = []

for q_no, (_, row) in enumerate(query_df.iterrows(), start=1):
    query = str(row["instruction"])

    # --------------------------------------------------------
    # BM25 TOP-1
    # --------------------------------------------------------
    tokenized_query = word_tokenize(
        query,
        engine="newmm",
    )

    bm25_scores = np.asarray(
        bm25.get_scores(tokenized_query),
        dtype=np.float64,
    )

    # เอกสารที่ใกล้/คะแนนสูงสุดเพียงตัวเดียว
    bm25_top_corpus_idx = int(np.argmax(bm25_scores))
    bm25_top_score = float(bm25_scores[bm25_top_corpus_idx])

    # ใช้ original dataset index เพื่อเอาไปเทียบกับ __index_level_0__
    bm25_top_index = corpus[bm25_top_corpus_idx].get(
        "__index_level_0__"
    )

    # --------------------------------------------------------
    # DENSE TOP-1
    # --------------------------------------------------------
    query_embedding = dense_model.encode(
        ["query: " + query],
        convert_to_numpy=True,
    )

    dense_scores = cosine_similarity(
        query_embedding,
        doc_embeddings,
    )[0]

    dense_top_corpus_idx = int(np.argmax(dense_scores))
    dense_top_score = float(dense_scores[dense_top_corpus_idx])

    dense_top_index = corpus[dense_top_corpus_idx].get(
        "__index_level_0__"
    )

    # --------------------------------------------------------
    # RRF
    #
    # ต้องรู้ "อันดับ" ภายในก่อนคำนวณ RRF
    # แต่ output จะคืนเฉพาะ index ของเอกสาร TOP-1
    # --------------------------------------------------------
    bm25_order = np.argsort(bm25_scores)[::-1]
    dense_order = np.argsort(dense_scores)[::-1]

    # array ที่ตำแหน่ง doc_idx เก็บ rank แบบ 1-based
    bm25_rank_by_doc = np.empty(n_docs, dtype=np.int32)
    bm25_rank_by_doc[bm25_order] = np.arange(
        1, n_docs + 1, dtype=np.int32
    )

    dense_rank_by_doc = np.empty(n_docs, dtype=np.int32)
    dense_rank_by_doc[dense_order] = np.arange(
        1, n_docs + 1, dtype=np.int32
    )

    rrf_scores = (
        1.0 / (RRF_K + bm25_rank_by_doc.astype(np.float64))
        +
        1.0 / (RRF_K + dense_rank_by_doc.astype(np.float64))
    )

    rrf_top_corpus_idx = int(np.argmax(rrf_scores))
    rrf_top_score = float(rrf_scores[rrf_top_corpus_idx])

    rrf_top_index = corpus[rrf_top_corpus_idx].get(
        "__index_level_0__"
    )

    result = row.to_dict()

    result.update({
        "BM25_score": bm25_top_score,
        "BM25_index": bm25_top_index,

        "DENSE_score": dense_top_score,
        "DENSE_index": dense_top_index,

        "RRF_score": rrf_top_score,
        "RRF_index": rrf_top_index,
    })

    results.append(result)

    print(
        f"Q{q_no:02d} | "
        f"BM25 index={bm25_top_index} score={bm25_top_score:.6f} | "
        f"DENSE index={dense_top_index} score={dense_top_score:.6f} | "
        f"RRF index={rrf_top_index} score={rrf_top_score:.8f}"
    )


# ============================================================
# SAVE
# ============================================================

out_df = pd.DataFrame(results)

original_cols = list(query_df.columns)

new_cols = [
    "BM25_score",
    "BM25_index",
    "DENSE_score",
    "DENSE_index",
    "RRF_score",
    "RRF_index",
]

out_df = out_df[original_cols + new_cols]

out_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print()
print("=" * 100)
print("TOP-1 RESULT")
print("=" * 100)

print(
    out_df[
        ["instruction"] + new_cols
    ].to_string(index=False)
)

print()
print(f"DONE -> {OUTPUT_FILE}")
