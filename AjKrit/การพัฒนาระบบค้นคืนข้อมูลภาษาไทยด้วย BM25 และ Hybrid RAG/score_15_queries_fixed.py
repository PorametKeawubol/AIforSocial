# -*- coding: utf-8 -*-
"""
เติม 6 columns ตามใบงาน:
BM25_score, BM25_rank, DENSE_score, DENSE_rank, RRF_score, RRF_rank

ใช้ไฟล์:
  thai_qa_utf8.json
  thai_qa_paraphrase_15.csv

ผลลัพธ์:
  thai_qa_paraphrase_15_scored.csv
"""

import sys
import subprocess
import json
from pathlib import Path


# ------------------------------------------------------------
# ติดตั้ง package ที่ขาดให้อัตโนมัติด้วย Python ตัวที่กำลังรัน
# ------------------------------------------------------------
PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pythainlp": "pythainlp",
    "rank_bm25": "rank-bm25",
    "sentence_transformers": "sentence-transformers",
    "sklearn": "scikit-learn",
}

missing = []
for module_name, pip_name in PACKAGES.items():
    try:
        __import__(module_name)
    except ImportError:
        missing.append(pip_name)

if missing:
    print("Installing missing packages:", ", ".join(missing))
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", *missing
    ])


import numpy as np
import pandas as pd
from pythainlp.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

def resolve_input_file(preferred_name, glob_pattern):
    preferred = BASE_DIR / preferred_name
    if preferred.exists():
        return preferred

    matches = sorted(BASE_DIR.glob(glob_pattern))
    if matches:
        print(f"    ไม่พบ {preferred_name} แต่พบไฟล์ใกล้เคียง -> {matches[0].name}")
        return matches[0]

    return preferred

JSON_FILE = resolve_input_file(
    "thai_qa_utf8.json",
    "thai_qa_utf8*.json",
)

CSV_FILE = resolve_input_file(
    "thai_qa_paraphrase_15.csv",
    "thai_qa_paraphrase_15*.csv",
)

MODEL_NAME = "intfloat/multilingual-e5-small"
RRF_K = 60

OUTPUT_FILE = BASE_DIR / "thai_qa_paraphrase_15_scored.csv"
EMBED_CACHE = BASE_DIR / "thai_instruction_e5_embeddings.npy"


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def norm_index(x):
    """normalize 2106, 2106.0, '2106' -> '2106'"""
    if pd.isna(x):
        return ""
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def rank_worksheet_style(scores):
    """
    ทำเหมือนตัวอย่างใบงาน:
        np.argsort(scores)[::-1]
    rank ที่ได้ในภายหลังเป็น 1-based
    """
    return np.argsort(scores)[::-1]


def inverse_rank_1based(ranking, n_docs):
    """
    ranking = doc indices เรียงอันดับ 1 -> N
    output[doc_idx] = rank แบบ 1-based
    """
    out = np.empty(n_docs, dtype=np.int32)
    out[ranking] = np.arange(1, n_docs + 1, dtype=np.int32)
    return out


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------
if not JSON_FILE.exists():
    nearby = [p.name for p in BASE_DIR.glob("*.json")]
    raise FileNotFoundError(
        f"ไม่พบไฟล์ JSON ที่ต้องใช้\n"
        f"โฟลเดอร์: {BASE_DIR}\n"
        f"ไฟล์ JSON ที่พบ: {nearby}"
    )

if not CSV_FILE.exists():
    nearby = [p.name for p in BASE_DIR.glob("*.csv")]
    raise FileNotFoundError(
        f"ไม่พบไฟล์ CSV ที่ต้องใช้\n"
        f"โฟลเดอร์: {BASE_DIR}\n"
        f"ไฟล์ CSV ที่พบ: {nearby}"
    )

print("[1] Loading corpus...")
with JSON_FILE.open("r", encoding="utf-8") as f:
    corpus = json.load(f)

test_df = pd.read_csv(CSV_FILE)

documents = [
    str(row.get("instruction", "") or "")
    for row in corpus
]

n_docs = len(documents)

print(f"    corpus = {n_docs:,} documents")
print(f"    queries = {len(test_df)}")


# ------------------------------------------------------------
# GROUND TRUTH LOOKUP
# ใช้ (instruction_org, __index_level_0__) เพราะ index อาจซ้ำข้าม source
# ------------------------------------------------------------
print("[2] Building ground-truth lookup...")

gt_lookup = {}

for corpus_idx, row in enumerate(corpus):
    key = (
        str(row.get("instruction", "") or "").strip(),
        norm_index(row.get("__index_level_0__"))
    )

    if key not in gt_lookup:
        gt_lookup[key] = corpus_idx


ground_truth_indices = []

for _, row in test_df.iterrows():
    key = (
        str(row["instruction_org"]).strip(),
        norm_index(row["__index_level_0__"])
    )

    if key not in gt_lookup:
        raise KeyError(
            "Ground truth not found:\n"
            f"instruction_org={key[0]}\n"
            f"index={key[1]}"
        )

    ground_truth_indices.append(gt_lookup[key])

print("    ground truth foundครบ:", len(ground_truth_indices))


# ------------------------------------------------------------
# BM25
# ตรงกับใบงาน: word_tokenize(..., engine="newmm")
# ไม่กรอง whitespace token เพิ่ม เพื่อให้ behavior ใกล้ snippet ที่สุด
# ------------------------------------------------------------
print("[3] Tokenizing Thai corpus with PyThaiNLP newmm...")
tokenized_docs = [
    word_tokenize(doc, engine="newmm")
    for doc in documents
]

print("[4] Building BM25Okapi...")
bm25 = BM25Okapi(tokenized_docs)


# ------------------------------------------------------------
# DENSE MODEL
# ------------------------------------------------------------
print(f"[5] Loading Dense model: {MODEL_NAME}")
dense_model = SentenceTransformer(MODEL_NAME)

if EMBED_CACHE.exists():
    print(f"    found cache: {EMBED_CACHE}")
    doc_embeddings = np.load(EMBED_CACHE)

    if (
        doc_embeddings.ndim != 2
        or doc_embeddings.shape[0] != n_docs
    ):
        print("    cache shape mismatch -> rebuilding")
        EMBED_CACHE.unlink()
        doc_embeddings = None
else:
    doc_embeddings = None

if doc_embeddings is None:
    print("[6] Encoding all documents...")
    print("    รอบแรกจะดาวน์โหลด model และใช้เวลาสร้าง embeddings")

    # ตาม E5 / โค้ดในใบงาน
    passages = ["passage: " + doc for doc in documents]

    doc_embeddings = dense_model.encode(
        passages,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    np.save(EMBED_CACHE, doc_embeddings)
    print(f"    saved cache -> {EMBED_CACHE}")
else:
    print(
        f"[6] Loaded embeddings cache: "
        f"{doc_embeddings.shape}"
    )


# ------------------------------------------------------------
# EVALUATE 15 QUERIES
# ------------------------------------------------------------
print("[7] Calculating BM25 / DENSE / RRF scores and ranks...")

BM25_scores_out = []
BM25_ranks_out = []

DENSE_scores_out = []
DENSE_ranks_out = []

RRF_scores_out = []
RRF_ranks_out = []


for q_i, (_, row) in enumerate(test_df.iterrows(), start=1):
    query = str(row["instruction"])
    gt_idx = ground_truth_indices[q_i - 1]

    # ========================================================
    # BM25
    # ========================================================
    tokenized_query = word_tokenize(
        query,
        engine="newmm"
    )

    bm25_scores = np.asarray(
        bm25.get_scores(tokenized_query),
        dtype=np.float64
    )

    bm25_ranking = rank_worksheet_style(bm25_scores)
    bm25_rank_by_doc = inverse_rank_1based(
        bm25_ranking,
        n_docs
    )

    gt_bm25_score = float(bm25_scores[gt_idx])
    gt_bm25_rank = int(bm25_rank_by_doc[gt_idx])

    # ========================================================
    # DENSE
    # ========================================================
    query_embedding = dense_model.encode(
        ["query: " + query],
        convert_to_numpy=True
    )

    dense_scores = cosine_similarity(
        query_embedding,
        doc_embeddings
    )[0]

    dense_ranking = rank_worksheet_style(dense_scores)
    dense_rank_by_doc = inverse_rank_1based(
        dense_ranking,
        n_docs
    )

    gt_dense_score = float(dense_scores[gt_idx])
    gt_dense_rank = int(dense_rank_by_doc[gt_idx])

    # ========================================================
    # RRF
    #
    # RRF(d) =
    #   1/(60 + BM25_rank(d))
    # + 1/(60 + Dense_rank(d))
    #
    # rank เป็น 1-based
    # ========================================================
    rrf_scores = (
        1.0 / (RRF_K + bm25_rank_by_doc.astype(np.float64))
        +
        1.0 / (RRF_K + dense_rank_by_doc.astype(np.float64))
    )

    rrf_ranking = rank_worksheet_style(rrf_scores)
    rrf_rank_by_doc = inverse_rank_1based(
        rrf_ranking,
        n_docs
    )

    gt_rrf_score = float(rrf_scores[gt_idx])
    gt_rrf_rank = int(rrf_rank_by_doc[gt_idx])

    BM25_scores_out.append(gt_bm25_score)
    BM25_ranks_out.append(gt_bm25_rank)

    DENSE_scores_out.append(gt_dense_score)
    DENSE_ranks_out.append(gt_dense_rank)

    RRF_scores_out.append(gt_rrf_score)
    RRF_ranks_out.append(gt_rrf_rank)

    print(
        f"Q{q_i:02d} | "
        f"BM25={gt_bm25_score:.6f} rank={gt_bm25_rank:<6} | "
        f"DENSE={gt_dense_score:.6f} rank={gt_dense_rank:<6} | "
        f"RRF={gt_rrf_score:.8f} rank={gt_rrf_rank}"
    )


# ------------------------------------------------------------
# SAVE EXACT 6 COLUMNS
# ------------------------------------------------------------
out = test_df.copy()

out["BM25_score"] = BM25_scores_out
out["BM25_rank"] = BM25_ranks_out

out["DENSE_score"] = DENSE_scores_out
out["DENSE_rank"] = DENSE_ranks_out

out["RRF_score"] = RRF_scores_out
out["RRF_rank"] = RRF_ranks_out


# จัด column ให้ตามที่ผู้ใช้ต้องการ
ordered_cols = list(test_df.columns) + [
    "BM25_score",
    "BM25_rank",
    "DENSE_score",
    "DENSE_rank",
    "RRF_score",
    "RRF_rank",
]

out = out[ordered_cols]

out.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print()
print("=" * 120)
print("FINAL TABLE")
print("=" * 120)

show_cols = [
    "BM25_score",
    "BM25_rank",
    "DENSE_score",
    "DENSE_rank",
    "RRF_score",
    "RRF_rank",
]

print(
    out[
        ["instruction"] + show_cols
    ].to_string(index=False)
)

print()
print(f"DONE -> {OUTPUT_FILE.resolve()}")
