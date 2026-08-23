# -*- coding: utf-8 -*-
"""
REAL TOP-1 INDEX VERSION

สิ่งที่ไฟล์นี้คืน:
    BM25_score, BM25_index
    DENSE_score, DENSE_index
    RRF_score, RRF_index

สำคัญ:
- *_index = ตำแหน่งจริงของเอกสารใน thai_qa_utf8.json (0..N-1)
- ไม่ใช่ rank
- ไม่ใช่ __index_level_0__ เพราะ field นั้นซ้ำกันได้หลาย source
- *_score = score ของเอกสาร TOP-1 ตัวนั้นจริง ๆ

ไฟล์ debug จะมีข้อความเอกสาร TOP-1 ให้ตรวจด้วยว่าเลือกอะไร
"""

import os
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

def resolve_file(preferred: str, pattern: str) -> Path:
    p = BASE_DIR / preferred
    if p.exists():
        return p

    matches = sorted(BASE_DIR.glob(pattern))
    if matches:
        print(f"ใช้ไฟล์ {matches[0].name}")
        return matches[0]

    raise FileNotFoundError(
        f"ไม่พบ {preferred} หรือไฟล์ที่ตรงกับ {pattern} ใน {BASE_DIR}"
    )


JSON_FILE = resolve_file("thai_qa_utf8.json", "thai_qa_utf8*.json")
CSV_FILE = resolve_file(
    "thai_qa_paraphrase_15.csv",
    "thai_qa_paraphrase_15*.csv"
)

MODEL_NAME = "intfloat/multilingual-e5-small"
RRF_K = 60

EMBED_CACHE = BASE_DIR / "thai_instruction_e5_embeddings.npy"

OUTPUT_FILE = BASE_DIR / "thai_qa_paraphrase_15_REAL_top1.csv"
DEBUG_FILE = BASE_DIR / "thai_qa_paraphrase_15_REAL_top1_debug.csv"


# ============================================================
# LOAD
# ============================================================

print("\n[1/6] Loading files...")

with JSON_FILE.open("r", encoding="utf-8") as f:
    corpus = json.load(f)

queries = pd.read_csv(CSV_FILE)

documents = [
    str(item.get("instruction", "") or "")
    for item in corpus
]

n_docs = len(documents)

print(f"Corpus size : {n_docs:,}")
print(f"Query count : {len(queries)}")


# ============================================================
# BM25
# ============================================================

print("\n[2/6] Thai tokenization (newmm)...")

tokenized_docs = [
    word_tokenize(doc, engine="newmm")
    for doc in documents
]

print("\n[3/6] Building BM25...")
bm25 = BM25Okapi(tokenized_docs)


# ============================================================
# DENSE
# ============================================================

print(f"\n[4/6] Loading {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

doc_embeddings = None

if EMBED_CACHE.exists():
    try:
        cached = np.load(EMBED_CACHE)

        if cached.ndim == 2 and cached.shape[0] == n_docs:
            doc_embeddings = cached
            print(
                f"ใช้ embedding cache เดิม: "
                f"{EMBED_CACHE.name} {cached.shape}"
            )
        else:
            print(
                f"cache shape {cached.shape} ไม่ตรงกับ corpus {n_docs:,}"
            )
    except Exception as e:
        print("อ่าน cache ไม่ได้:", e)


if doc_embeddings is None:
    print("\n[5/6] Encoding corpus...")

    doc_embeddings = model.encode(
        ["passage: " + doc for doc in documents],
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    np.save(EMBED_CACHE, doc_embeddings)
    print(f"Saved cache -> {EMBED_CACHE.name}")
else:
    print("\n[5/6] Skip encoding เพราะมี cache แล้ว")


# ============================================================
# SEARCH
# ============================================================

print("\n[6/6] REAL TOP-1 search...\n")

output_rows = []
debug_rows = []


for q_no, (_, row) in enumerate(queries.iterrows(), start=1):

    query = str(row["instruction"])

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    q_tokens = word_tokenize(query, engine="newmm")

    bm25_scores = np.asarray(
        bm25.get_scores(q_tokens),
        dtype=np.float64
    )

    # นี่คือ INDEX จริงของเอกสารที่คะแนนสูงสุด
    BM25_index = int(np.argmax(bm25_scores))
    BM25_score = float(bm25_scores[BM25_index])


    # --------------------------------------------------------
    # DENSE
    # --------------------------------------------------------

    q_embedding = model.encode(
        ["query: " + query],
        convert_to_numpy=True
    )

    dense_scores = cosine_similarity(
        q_embedding,
        doc_embeddings
    )[0]

    # INDEX จริงของเอกสาร semantic ที่ใกล้ที่สุด
    DENSE_index = int(np.argmax(dense_scores))
    DENSE_score = float(dense_scores[DENSE_index])


    # --------------------------------------------------------
    # RRF
    # --------------------------------------------------------

    # RRF จำเป็นต้องสร้าง rank ภายใน
    # แต่ output ที่คืนคือ INDEX ของเอกสารที่ RRF สูงสุด

    bm25_order = np.argsort(bm25_scores)[::-1]
    dense_order = np.argsort(dense_scores)[::-1]

    bm25_rank = np.empty(n_docs, dtype=np.int32)
    dense_rank = np.empty(n_docs, dtype=np.int32)

    bm25_rank[bm25_order] = np.arange(
        1, n_docs + 1, dtype=np.int32
    )

    dense_rank[dense_order] = np.arange(
        1, n_docs + 1, dtype=np.int32
    )

    rrf_scores = (
        1.0 / (RRF_K + bm25_rank)
        +
        1.0 / (RRF_K + dense_rank)
    )

    # INDEX จริงของเอกสาร fusion ที่ใกล้ที่สุด
    RRF_index = int(np.argmax(rrf_scores))
    RRF_score = float(rrf_scores[RRF_index])


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    base = row.to_dict()

    result = {
        **base,
        "BM25_score": BM25_score,
        "BM25_index": BM25_index,
        "DENSE_score": DENSE_score,
        "DENSE_index": DENSE_index,
        "RRF_score": RRF_score,
        "RRF_index": RRF_index,
    }

    output_rows.append(result)


    # debug เพื่อพิสูจน์ว่า index ชี้ไปยังเอกสารอะไรจริง
    debug_rows.append({
        **result,

        "BM25_original_index": corpus[BM25_index].get(
            "__index_level_0__"
        ),
        "BM25_doc": corpus[BM25_index].get("instruction", ""),
        "BM25_answer": corpus[BM25_index].get("answer", ""),

        "DENSE_original_index": corpus[DENSE_index].get(
            "__index_level_0__"
        ),
        "DENSE_doc": corpus[DENSE_index].get("instruction", ""),
        "DENSE_answer": corpus[DENSE_index].get("answer", ""),

        "RRF_original_index": corpus[RRF_index].get(
            "__index_level_0__"
        ),
        "RRF_doc": corpus[RRF_index].get("instruction", ""),
        "RRF_answer": corpus[RRF_index].get("answer", ""),
    })


    print(
        f"Q{q_no:02d}\n"
        f"  BM25 : score={BM25_score:.8f} "
        f"index={BM25_index} | "
        f"{documents[BM25_index][:80]}\n"
        f"  DENSE: score={DENSE_score:.8f} "
        f"index={DENSE_index} | "
        f"{documents[DENSE_index][:80]}\n"
        f"  RRF  : score={RRF_score:.8f} "
        f"index={RRF_index} | "
        f"{documents[RRF_index][:80]}\n"
    )


# ============================================================
# SAVE
# ============================================================

result_df = pd.DataFrame(output_rows)

wanted_columns = list(queries.columns) + [
    "BM25_score",
    "BM25_index",
    "DENSE_score",
    "DENSE_index",
    "RRF_score",
    "RRF_index",
]

result_df = result_df[wanted_columns]

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(debug_rows).to_csv(
    DEBUG_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("=" * 100)
print("FINAL")
print("=" * 100)

print(
    result_df[
        [
            "instruction",
            "BM25_score",
            "BM25_index",
            "DENSE_score",
            "DENSE_index",
            "RRF_score",
            "RRF_index",
        ]
    ].to_string(index=False)
)

print()
print(f"RESULT -> {OUTPUT_FILE}")
print(f"DEBUG  -> {DEBUG_FILE}")
print()
print(
    "เปิด DEBUG CSV เพื่อตรวจว่าแต่ละ index "
    "ชี้ไปยัง document ไหนจริง ๆ"
)
