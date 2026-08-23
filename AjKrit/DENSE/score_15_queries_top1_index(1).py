# -*- coding: utf-8 -*-

"""
Thai Retrieval Evaluation
=========================

เปรียบเทียบ 3 วิธี:
1. BM25
2. Dense Vector Search (multilingual-e5-small)
3. Hybrid Search ด้วย Reciprocal Rank Fusion (RRF)

Dataset:
- thai_qa_utf8.json
- thai_qa_paraphrase_15.csv

Output:
- thai_qa_paraphrase_15_top1_scored.csv

หมายเหตุ:
- Corpus ใช้ field "instruction" จาก JSON
- Query ใช้ field "instruction" จาก CSV (paraphrased query)
- Ground Truth หาโดยใช้:
    __index_level_0__
    + instruction_org
    + answer
  เพื่อระบุตัว document จริงใน corpus
"""

# ============================================================
# IMPORTS
# ============================================================

import os

# ไม่ใช้ TensorFlow
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
    """
    ถ้ามีชื่อไฟล์ตรงตาม preferred_name ให้ใช้ไฟล์นั้น
    ถ้าไม่มี ให้หาไฟล์จาก wildcard pattern
    """

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


# Cache embedding
EMBED_CACHE = (
    BASE_DIR
    / "thai_instruction_e5_embeddings.npy"
)


# Output CSV
OUTPUT_FILE = (
    BASE_DIR
    / "thai_qa_paraphrase_15_top1_scored.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_text(value):
    """
    แปลงค่าเป็น string อย่างปลอดภัย
    """

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_index(value):
    """
    ทำให้ index จาก JSON และ CSV เทียบกันได้

    เช่น:
    2106
    2106.0
    "2106"

    จะกลายเป็น 2106 เหมือนกัน
    """

    if value is None:
        return None

    if pd.isna(value):
        return None

    try:
        return int(float(value))
    except (ValueError, TypeError):
        return safe_text(value)


# ============================================================
# CHECK FILES
# ============================================================

if not JSON_FILE.exists():

    raise FileNotFoundError(
        f"ไม่พบไฟล์ thai_qa_utf8*.json ใน {BASE_DIR}"
    )


if not CSV_FILE.exists():

    raise FileNotFoundError(
        f"ไม่พบไฟล์ thai_qa_paraphrase_15*.csv "
        f"ใน {BASE_DIR}"
    )


# ============================================================
# 1. LOAD DATA
# ============================================================

print()
print("=" * 100)
print("THAI RETRIEVAL EVALUATION")
print("=" * 100)

print()
print("[1/7] Loading data...")


with JSON_FILE.open(
    "r",
    encoding="utf-8",
) as f:

    corpus = json.load(f)


query_df = pd.read_csv(
    CSV_FILE
)


# Corpus ใช้ instruction
documents = [

    safe_text(
        row.get("instruction", "")
    )

    for row in corpus
]


n_docs = len(documents)
n_queries = len(query_df)


print(
    f"  Corpus : {n_docs:,}"
)

print(
    f"  Queries: {n_queries}"
)


# ============================================================
# 2. BUILD GROUND TRUTH LOOKUP
# ============================================================

print()
print(
    "[2/7] Building Ground Truth lookup..."
)


"""
เราไม่ใช้ __index_level_0__ เพียงตัวเดียวในการหา Ground Truth

ใช้ key:

(
    __index_level_0__,
    instruction,
    answer
)

โดย instruction ใน corpus
ต้องตรงกับ instruction_org ใน CSV
"""


gt_lookup = {}


for corpus_idx, row in enumerate(corpus):

    key = (

        normalize_index(
            row.get("__index_level_0__")
        ),

        safe_text(
            row.get("instruction")
        ),

        safe_text(
            row.get("answer")
        ),
    )

    if key not in gt_lookup:

        gt_lookup[key] = []

    gt_lookup[key].append(
        corpus_idx
    )


def get_ground_truth_corpus_idx(row):
    """
    คืนตำแหน่ง Ground Truth จริงภายใน corpus
    """

    key = (

        normalize_index(
            row["__index_level_0__"]
        ),

        safe_text(
            row["instruction_org"]
        ),

        safe_text(
            row["answer"]
        ),
    )

    matches = gt_lookup.get(
        key,
        [],
    )

    if len(matches) == 1:

        return matches[0]


    # --------------------------------------------------------
    # Fallback:
    # ถ้าจับครบ 3 field ไม่เจอ
    # ลองหา index + instruction_org
    # --------------------------------------------------------

    target_index = normalize_index(
        row["__index_level_0__"]
    )

    target_instruction = safe_text(
        row["instruction_org"]
    )

    candidate_matches = []


    for corpus_idx, corpus_row in enumerate(corpus):

        corpus_original_index = normalize_index(
            corpus_row.get("__index_level_0__")
        )

        corpus_instruction = safe_text(
            corpus_row.get("instruction")
        )

        if (
            corpus_original_index == target_index
            and
            corpus_instruction == target_instruction
        ):

            candidate_matches.append(
                corpus_idx
            )


    if len(candidate_matches) == 1:

        return candidate_matches[0]


    raise ValueError(

        "\nไม่สามารถหา Ground Truth ได้แน่นอน\n"

        f"instruction_org = "
        f"{target_instruction}\n"

        f"__index_level_0__ = "
        f"{target_index}\n"

        f"matches = "
        f"{len(matches)}\n"

        f"fallback matches = "
        f"{len(candidate_matches)}"
    )


# เช็ก Ground Truth ทั้ง 15 query ก่อน
ground_truth_indices = []


for _, row in query_df.iterrows():

    gt_idx = get_ground_truth_corpus_idx(
        row
    )

    ground_truth_indices.append(
        gt_idx
    )


print(
    f"  Ground Truth resolved: "
    f"{len(ground_truth_indices)}/{n_queries}"
)


# ============================================================
# 3. BM25
# ============================================================

print()
print(
    "[3/7] Thai tokenization..."
)


tokenized_docs = [

    word_tokenize(
        doc,
        engine="newmm",
    )

    for doc in documents
]


print()
print(
    "[4/7] Building BM25..."
)


bm25 = BM25Okapi(
    tokenized_docs
)


# ============================================================
# 4. DENSE MODEL
# ============================================================

print()
print(
    f"[5/7] Loading Dense model: {MODEL_NAME}"
)


dense_model = SentenceTransformer(
    MODEL_NAME
)


doc_embeddings = None


# ============================================================
# LOAD CACHE
# ============================================================

if EMBED_CACHE.exists():

    try:

        cached = np.load(
            EMBED_CACHE
        )

        if (
            cached.ndim == 2
            and
            cached.shape[0] == n_docs
        ):

            doc_embeddings = cached

            print(
                f"  Loaded cache: "
                f"{EMBED_CACHE.name}"
            )

            print(
                f"  Shape: "
                f"{doc_embeddings.shape}"
            )

        else:

            print(
                "  Embedding cache shape "
                "ไม่ตรงกับ corpus"
            )

            print(
                "  -> จะสร้าง embeddings ใหม่"
            )

    except Exception as e:

        print(
            f"  อ่าน embedding cache "
            f"ไม่สำเร็จ: {e}"
        )


# ============================================================
# CREATE EMBEDDINGS IF NO CACHE
# ============================================================

if doc_embeddings is None:

    print()
    print(
        "[6/7] Encoding corpus embeddings..."
    )


    # E5 document prefix
    passages = [

        "passage: " + doc

        for doc in documents
    ]


    doc_embeddings = dense_model.encode(

        passages,

        batch_size=128,

        show_progress_bar=True,

        convert_to_numpy=True,
    )


    np.save(

        EMBED_CACHE,

        doc_embeddings,
    )


    print(
        f"  Saved cache: "
        f"{EMBED_CACHE.name}"
    )


else:

    print()

    print(
        "[6/7] Skip encoding "
        "(using embedding cache)"
    )


# ============================================================
# 5. SEARCH ALL QUERIES
# ============================================================

print()
print(
    "[7/7] Searching 15 queries..."
)

print()


results = []


for q_no, (_, row) in enumerate(
    query_df.iterrows(),
    start=1,
):

    # ========================================================
    # QUERY
    # ========================================================

    query = safe_text(
        row["instruction"]
    )


    # Ground Truth ตำแหน่งจริงใน corpus
    gt_corpus_idx = (
        ground_truth_indices[q_no - 1]
    )


    gt_original_index = normalize_index(
        row["__index_level_0__"]
    )


    # ========================================================
    # BM25
    # ========================================================

    tokenized_query = word_tokenize(

        query,

        engine="newmm",
    )


    bm25_scores = np.asarray(

        bm25.get_scores(
            tokenized_query
        ),

        dtype=np.float64,
    )


    # Ranking จากมาก -> น้อย
    bm25_order = np.argsort(
        bm25_scores
    )[::-1]


    bm25_top_corpus_idx = int(
        bm25_order[0]
    )


    bm25_top_score = float(
        bm25_scores[
            bm25_top_corpus_idx
        ]
    )


    bm25_top_index = normalize_index(

        corpus[
            bm25_top_corpus_idx
        ].get(
            "__index_level_0__"
        )
    )


    # ตรวจ Top-1 ว่าเป็น document เดียวกับ Ground Truth จริงไหม
    bm25_correct = (

        bm25_top_corpus_idx
        ==
        gt_corpus_idx
    )


    # ========================================================
    # DENSE VECTOR SEARCH
    # ========================================================

    # E5 query prefix
    query_embedding = dense_model.encode(

        [
            "query: " + query
        ],

        convert_to_numpy=True,
    )


    dense_scores = cosine_similarity(

        query_embedding,

        doc_embeddings,

    )[0]


    dense_order = np.argsort(
        dense_scores
    )[::-1]


    dense_top_corpus_idx = int(
        dense_order[0]
    )


    dense_top_score = float(

        dense_scores[
            dense_top_corpus_idx
        ]
    )


    dense_top_index = normalize_index(

        corpus[
            dense_top_corpus_idx
        ].get(
            "__index_level_0__"
        )
    )


    dense_correct = (

        dense_top_corpus_idx
        ==
        gt_corpus_idx
    )


    # ========================================================
    # RRF
    # ========================================================

    """
    Reciprocal Rank Fusion

    RRF(d) =
        1 / (k + BM25_rank)
        +
        1 / (k + Dense_rank)

    k = 60
    """


    # --------------------------------------------------------
    # BM25 rank ของแต่ละ document
    # --------------------------------------------------------

    bm25_rank_by_doc = np.empty(

        n_docs,

        dtype=np.int32,
    )


    bm25_rank_by_doc[
        bm25_order
    ] = np.arange(

        1,

        n_docs + 1,

        dtype=np.int32,
    )


    # --------------------------------------------------------
    # Dense rank ของแต่ละ document
    # --------------------------------------------------------

    dense_rank_by_doc = np.empty(

        n_docs,

        dtype=np.int32,
    )


    dense_rank_by_doc[
        dense_order
    ] = np.arange(

        1,

        n_docs + 1,

        dtype=np.int32,
    )


    # --------------------------------------------------------
    # RRF score
    # --------------------------------------------------------

    rrf_scores = (

        1.0
        /
        (
            RRF_K
            +
            bm25_rank_by_doc.astype(
                np.float64
            )
        )

        +

        1.0
        /
        (
            RRF_K
            +
            dense_rank_by_doc.astype(
                np.float64
            )
        )
    )


    rrf_order = np.argsort(
        rrf_scores
    )[::-1]


    rrf_top_corpus_idx = int(
        rrf_order[0]
    )


    rrf_top_score = float(

        rrf_scores[
            rrf_top_corpus_idx
        ]
    )


    rrf_top_index = normalize_index(

        corpus[
            rrf_top_corpus_idx
        ].get(
            "__index_level_0__"
        )
    )


    rrf_correct = (

        rrf_top_corpus_idx
        ==
        gt_corpus_idx
    )


    # ========================================================
    # GROUND TRUTH SCORES / RANKS
    # ========================================================

    # Rank ของ Ground Truth ใน BM25
    bm25_gt_rank = int(
        bm25_rank_by_doc[
            gt_corpus_idx
        ]
    )


    # Rank ของ Ground Truth ใน Dense
    dense_gt_rank = int(
        dense_rank_by_doc[
            gt_corpus_idx
        ]
    )


    # RRF ranking
    rrf_rank_by_doc = np.empty(

        n_docs,

        dtype=np.int32,
    )


    rrf_rank_by_doc[
        rrf_order
    ] = np.arange(

        1,

        n_docs + 1,

        dtype=np.int32,
    )


    rrf_gt_rank = int(

        rrf_rank_by_doc[
            gt_corpus_idx
        ]
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    result = row.to_dict()


    result.update({

        # ----------------------------------------------------
        # Ground Truth
        # ----------------------------------------------------

        "GT_corpus_idx":
            gt_corpus_idx,

        "GT_index":
            gt_original_index,


        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        "BM25_score":
            bm25_top_score,

        "BM25_index":
            bm25_top_index,

        "BM25_GT_rank":
            bm25_gt_rank,

        "BM25_correct":
            bm25_correct,


        # ----------------------------------------------------
        # Dense
        # ----------------------------------------------------

        "DENSE_score":
            dense_top_score,

        "DENSE_index":
            dense_top_index,

        "DENSE_GT_rank":
            dense_gt_rank,

        "DENSE_correct":
            dense_correct,


        # ----------------------------------------------------
        # RRF
        # ----------------------------------------------------

        "RRF_score":
            rrf_top_score,

        "RRF_index":
            rrf_top_index,

        "RRF_GT_rank":
            rrf_gt_rank,

        "RRF_correct":
            rrf_correct,
    })


    results.append(
        result
    )


    # ========================================================
    # PRINT EACH QUERY
    # ========================================================

    print(
        f"Q{q_no:02d} | "
        f"GT={gt_original_index} | "
        f"BM25={bm25_top_index} "
        f"(rankGT={bm25_gt_rank}, "
        f"{'✓' if bm25_correct else '✗'}) | "
        f"DENSE={dense_top_index} "
        f"(rankGT={dense_gt_rank}, "
        f"{'✓' if dense_correct else '✗'}) | "
        f"RRF={rrf_top_index} "
        f"(rankGT={rrf_gt_rank}, "
        f"{'✓' if rrf_correct else '✗'})"
    )


# ============================================================
# 6. CREATE OUTPUT DATAFRAME
# ============================================================

out_df = pd.DataFrame(
    results
)


original_cols = list(
    query_df.columns
)


new_cols = [

    # Ground Truth
    "GT_corpus_idx",
    "GT_index",

    # BM25
    "BM25_score",
    "BM25_index",
    "BM25_GT_rank",
    "BM25_correct",

    # Dense
    "DENSE_score",
    "DENSE_index",
    "DENSE_GT_rank",
    "DENSE_correct",

    # RRF
    "RRF_score",
    "RRF_index",
    "RRF_GT_rank",
    "RRF_correct",
]


out_df = out_df[
    original_cols
    +
    new_cols
]


# ============================================================
# 7. SAVE CSV
# ============================================================

out_df.to_csv(

    OUTPUT_FILE,

    index=False,

    encoding="utf-8-sig",
)


# ============================================================
# 8. SUMMARY
# ============================================================

bm25_correct_count = int(
    out_df[
        "BM25_correct"
    ].sum()
)


dense_correct_count = int(
    out_df[
        "DENSE_correct"
    ].sum()
)


rrf_correct_count = int(
    out_df[
        "RRF_correct"
    ].sum()
)


total = len(
    out_df
)


bm25_accuracy = (
    bm25_correct_count
    /
    total
)


dense_accuracy = (
    dense_correct_count
    /
    total
)


rrf_accuracy = (
    rrf_correct_count
    /
    total
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 100)

print(
    "TOP-1 RESULT"
)

print("=" * 100)


display_cols = [

    "instruction",

    "GT_index",

    "BM25_index",
    "BM25_GT_rank",
    "BM25_correct",

    "DENSE_index",
    "DENSE_GT_rank",
    "DENSE_correct",

    "RRF_index",
    "RRF_GT_rank",
    "RRF_correct",
]


print(

    out_df[
        display_cols
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)

print(
    "TOP-1 ACCURACY"
)

print("=" * 100)


print(
    f"BM25  : "
    f"{bm25_correct_count}/{total} "
    f"= {bm25_accuracy:.2%}"
)


print(
    f"DENSE : "
    f"{dense_correct_count}/{total} "
    f"= {dense_accuracy:.2%}"
)


print(
    f"RRF   : "
    f"{rrf_correct_count}/{total} "
    f"= {rrf_accuracy:.2%}"
)


print()
print("=" * 100)

print(
    f"DONE -> {OUTPUT_FILE}"
)

print("=" * 100)