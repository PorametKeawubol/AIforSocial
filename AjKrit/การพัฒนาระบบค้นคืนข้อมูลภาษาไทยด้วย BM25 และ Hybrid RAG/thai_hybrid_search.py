# -*- coding: utf-8 -*-
"""
Thai Hybrid Search Lab
BM25 + TF-IDF + Dense Search + Reciprocal Rank Fusion (RRF)

ไฟล์ที่ใช้:
- thai_qa_utf8.json
- thai_qa_paraphrase_15.csv

ผลลัพธ์:
- hybrid_search_results.csv
- hybrid_search_summary.csv

Run:
    python thai_hybrid_search.py

หรือกำหนดไฟล์เอง:
    python thai_hybrid_search.py --json thai_qa_utf8.json --csv thai_qa_paraphrase_15.csv
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pythainlp.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIG
# ============================================================

DEFAULT_JSON = "thai_qa_utf8.json"
DEFAULT_CSV = "thai_qa_paraphrase_15.csv"

DENSE_MODEL_NAME = "intfloat/multilingual-e5-small"
RRF_K = 60
BATCH_SIZE = 128

DETAIL_OUTPUT = "hybrid_search_results.csv"
SUMMARY_OUTPUT = "hybrid_search_summary.csv"
EMBEDDING_CACHE = "thai_instruction_embeddings.npy"


# ============================================================
# 1. THAI TOKENIZATION
# ============================================================

def thai_tokenize(text):
    """
    ตัดคำภาษาไทยด้วย PyThaiNLP newmm
    และตัด token ว่างออก
    """
    text = "" if text is None else str(text)
    return [
        token.strip()
        for token in word_tokenize(text, engine="newmm")
        if token.strip()
    ]


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data(json_path, csv_path):
    print("\n[1/7] Loading dataset...")

    with open(json_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    queries = pd.read_csv(csv_path)

    required_csv_columns = {
        "instruction_org",
        "instruction",
        "answer",
        "__index_level_0__",
    }

    missing = required_csv_columns - set(queries.columns)
    if missing:
        raise ValueError(
            f"CSV ขาด column: {sorted(missing)}"
        )

    documents = [
        str(row.get("instruction", "") or "")
        for row in corpus
    ]

    print(f"  Corpus documents : {len(documents):,}")
    print(f"  Test queries     : {len(queries):,}")

    return corpus, documents, queries


# ============================================================
# 3. FIND GROUND-TRUTH DOCUMENT
# ============================================================

def normalize_index_value(value):
    """
    ทำให้ค่า __index_level_0__ เปรียบเทียบกันได้
    เช่น 2106, 2106.0, '2106'
    """
    if pd.isna(value):
        return None

    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return str(value).strip()


def build_ground_truth_lookup(corpus):
    """
    ใช้คู่:
        (instruction, __index_level_0__)
    เพื่อระบุตำแหน่งเอกสารที่ถูกต้องใน JSON

    เหตุผล:
    __index_level_0__ อาจซ้ำกันข้าม source ได้
    จึงไม่ควรใช้ index field อย่างเดียว
    """
    lookup = {}

    for corpus_idx, row in enumerate(corpus):
        instruction = str(row.get("instruction", "") or "").strip()
        original_idx = normalize_index_value(row.get("__index_level_0__"))

        key = (instruction, original_idx)

        if key not in lookup:
            lookup[key] = corpus_idx

    return lookup


def get_ground_truth_index(test_row, gt_lookup):
    instruction_org = str(test_row["instruction_org"]).strip()
    original_idx = normalize_index_value(test_row["__index_level_0__"])

    key = (instruction_org, original_idx)

    if key not in gt_lookup:
        raise KeyError(
            "หา Ground Truth ไม่พบใน JSON:\n"
            f"  instruction_org = {instruction_org}\n"
            f"  __index_level_0__ = {original_idx}"
        )

    return gt_lookup[key]


# ============================================================
# 4. LEXICAL SEARCH: TF-IDF + BM25
# ============================================================

def build_lexical_models(documents):
    print("\n[2/7] Thai tokenization + building BM25...")

    start = time.time()

    tokenized_docs = [thai_tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    elapsed = time.time() - start
    print(f"  BM25 ready in {elapsed:.2f} sec")

    print("\n[3/7] Building TF-IDF baseline...")

    start = time.time()

    # tokenizer=thai_tokenize ทำให้ TF-IDF ใช้การตัดคำภาษาไทยชุดเดียวกับ BM25
    tfidf = TfidfVectorizer(
        tokenizer=thai_tokenize,
        token_pattern=None,
        lowercase=False,
        norm="l2",
    )

    tfidf_matrix = tfidf.fit_transform(documents)

    elapsed = time.time() - start
    print(
        f"  TF-IDF ready: matrix={tfidf_matrix.shape}, "
        f"time={elapsed:.2f} sec"
    )

    return tokenized_docs, bm25, tfidf, tfidf_matrix


# ============================================================
# 5. DENSE / SEMANTIC SEARCH
# ============================================================

def build_dense_model_and_embeddings(
    documents,
    model_name=DENSE_MODEL_NAME,
    cache_path=EMBEDDING_CACHE,
    batch_size=BATCH_SIZE,
):
    print(f"\n[4/7] Loading Dense model: {model_name}")
    model = SentenceTransformer(model_name)

    cache_path = Path(cache_path)

    # ถ้ามี cache และจำนวน document ตรงกัน ให้ใช้ของเดิม
    if cache_path.exists():
        try:
            embeddings = np.load(cache_path)

            if embeddings.shape[0] == len(documents):
                print(
                    f"  Loaded cached embeddings: "
                    f"{cache_path} {embeddings.shape}"
                )
                return model, embeddings

            print(
                "  Cache document count ไม่ตรงกับ corpus "
                "จึงสร้าง embeddings ใหม่"
            )
        except Exception as e:
            print(f"  โหลด cache ไม่สำเร็จ: {e}")

    print("  Encoding corpus embeddings...")
    start = time.time()

    # E5 ต้องใส่ passage: ให้เอกสาร
    passages = ["passage: " + doc for doc in documents]

    embeddings = model.encode(
        passages,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    np.save(cache_path, embeddings)

    elapsed = time.time() - start
    print(
        f"  Dense embeddings ready: {embeddings.shape}, "
        f"time={elapsed:.2f} sec"
    )
    print(f"  Saved cache -> {cache_path}")

    return model, embeddings


# ============================================================
# 6. SEARCH FUNCTIONS
# ============================================================

def rank_from_scores(scores):
    """
    คืนค่า index เอกสารเรียงคะแนนมาก -> น้อย
    """
    return np.argsort(-np.asarray(scores), kind="stable")


def inverse_ranks(ranking, n_docs):
    """
    ranking = [doc_idx ของ rank 1, rank 2, ...]
    คืน array ที่บอกว่าเอกสารแต่ละตัวอยู่ rank ที่เท่าไร (0-based)
    """
    inv = np.empty(n_docs, dtype=np.int32)
    inv[ranking] = np.arange(n_docs, dtype=np.int32)
    return inv


def calculate_rrf_scores(
    bm25_ranking,
    dense_ranking,
    n_docs,
    k=RRF_K,
):
    """
    Reciprocal Rank Fusion:

        RRF(d) = 1/(k + rank_bm25)
               + 1/(k + rank_dense)

    ในโค้ด rank เป็น 1-based ตามสูตร
    """
    bm25_inv = inverse_ranks(bm25_ranking, n_docs)
    dense_inv = inverse_ranks(dense_ranking, n_docs)

    bm25_rank_1based = bm25_inv.astype(np.float64) + 1.0
    dense_rank_1based = dense_inv.astype(np.float64) + 1.0

    rrf_scores = (
        1.0 / (k + bm25_rank_1based)
        + 1.0 / (k + dense_rank_1based)
    )

    return rrf_scores


def get_rank_of_doc(ranking, doc_idx):
    """
    คืน rank แบบ 1-based
    """
    positions = np.where(ranking == doc_idx)[0]

    if len(positions) == 0:
        return None

    return int(positions[0]) + 1


def search_all_methods(
    query,
    bm25,
    tfidf,
    tfidf_matrix,
    dense_model,
    dense_embeddings,
    rrf_k=RRF_K,
):
    # ---------- TF-IDF ----------
    query_tfidf = tfidf.transform([query])

    # เอกสารถูก normalize L2 อยู่แล้ว
    # ดังนั้น dot product = cosine similarity
    tfidf_scores = (
        tfidf_matrix @ query_tfidf.T
    ).toarray().ravel()

    tfidf_ranking = rank_from_scores(tfidf_scores)

    # ---------- BM25 ----------
    tokenized_query = thai_tokenize(query)
    bm25_scores = np.asarray(
        bm25.get_scores(tokenized_query),
        dtype=np.float64,
    )

    bm25_ranking = rank_from_scores(bm25_scores)

    # ---------- Dense ----------
    # E5 ต้องใส่ query:
    query_embedding = dense_model.encode(
        ["query: " + query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype("float32")

    # corpus + query normalize แล้ว:
    # dot product = cosine similarity
    dense_scores = dense_embeddings @ query_embedding
    dense_ranking = rank_from_scores(dense_scores)

    # ---------- Hybrid RRF ----------
    rrf_scores = calculate_rrf_scores(
        bm25_ranking=bm25_ranking,
        dense_ranking=dense_ranking,
        n_docs=len(dense_embeddings),
        k=rrf_k,
    )

    rrf_ranking = rank_from_scores(rrf_scores)

    return {
        "tfidf_scores": tfidf_scores,
        "tfidf_ranking": tfidf_ranking,
        "bm25_scores": bm25_scores,
        "bm25_ranking": bm25_ranking,
        "dense_scores": dense_scores,
        "dense_ranking": dense_ranking,
        "rrf_scores": rrf_scores,
        "rrf_ranking": rrf_ranking,
    }


# ============================================================
# 7. EVALUATION
# ============================================================

def safe_text(value):
    if value is None:
        return ""
    return str(value)


def evaluate(
    corpus,
    queries,
    gt_lookup,
    bm25,
    tfidf,
    tfidf_matrix,
    dense_model,
    dense_embeddings,
):
    print("\n[5/7] Evaluating 15 paraphrase queries...")

    records = []

    for q_no, (_, row) in enumerate(queries.iterrows(), start=1):
        query = safe_text(row["instruction"]).strip()

        gt_idx = get_ground_truth_index(row, gt_lookup)
        gt_doc = safe_text(corpus[gt_idx].get("instruction"))
        gt_answer = safe_text(corpus[gt_idx].get("answer"))

        result = search_all_methods(
            query=query,
            bm25=bm25,
            tfidf=tfidf,
            tfidf_matrix=tfidf_matrix,
            dense_model=dense_model,
            dense_embeddings=dense_embeddings,
        )

        tfidf_rank = get_rank_of_doc(
            result["tfidf_ranking"], gt_idx
        )
        bm25_rank = get_rank_of_doc(
            result["bm25_ranking"], gt_idx
        )
        dense_rank = get_rank_of_doc(
            result["dense_ranking"], gt_idx
        )
        rrf_rank = get_rank_of_doc(
            result["rrf_ranking"], gt_idx
        )

        tfidf_top_idx = int(result["tfidf_ranking"][0])
        bm25_top_idx = int(result["bm25_ranking"][0])
        dense_top_idx = int(result["dense_ranking"][0])
        rrf_top_idx = int(result["rrf_ranking"][0])

        record = {
            "query_no": q_no,
            "instruction_org": safe_text(row["instruction_org"]),
            "query_paraphrase": query,
            "expected_answer": safe_text(row["answer"]),
            "ground_truth_corpus_index": gt_idx,
            "ground_truth_original_index": row["__index_level_0__"],

            # TF-IDF
            "TFIDF_score": float(result["tfidf_scores"][gt_idx]),
            "TFIDF_rank": tfidf_rank,
            "TFIDF_top1_doc": safe_text(
                corpus[tfidf_top_idx].get("instruction")
            ),
            "TFIDF_top1_answer": safe_text(
                corpus[tfidf_top_idx].get("answer")
            ),

            # BM25
            "BM25_score": float(result["bm25_scores"][gt_idx]),
            "BM25_rank": bm25_rank,
            "BM25_top1_doc": safe_text(
                corpus[bm25_top_idx].get("instruction")
            ),
            "BM25_top1_answer": safe_text(
                corpus[bm25_top_idx].get("answer")
            ),

            # Dense
            "DENSE_score": float(result["dense_scores"][gt_idx]),
            "DENSE_rank": dense_rank,
            "DENSE_top1_doc": safe_text(
                corpus[dense_top_idx].get("instruction")
            ),
            "DENSE_top1_answer": safe_text(
                corpus[dense_top_idx].get("answer")
            ),

            # Hybrid RRF
            "RRF_score": float(result["rrf_scores"][gt_idx]),
            "RRF_rank": rrf_rank,
            "RRF_top1_doc": safe_text(
                corpus[rrf_top_idx].get("instruction")
            ),
            "RRF_top1_answer": safe_text(
                corpus[rrf_top_idx].get("answer")
            ),

            # ใช้เช็คง่าย
            "TFIDF_hit1": int(tfidf_rank == 1),
            "BM25_hit1": int(bm25_rank == 1),
            "DENSE_hit1": int(dense_rank == 1),
            "RRF_hit1": int(rrf_rank == 1),

            "ground_truth_doc": gt_doc,
            "ground_truth_answer": gt_answer,
        }

        records.append(record)

        print(
            f"  Q{q_no:02d} | "
            f"BM25 rank={bm25_rank:<5} "
            f"score={record['BM25_score']:.4f} | "
            f"DENSE rank={dense_rank:<5} "
            f"score={record['DENSE_score']:.4f} | "
            f"RRF rank={rrf_rank:<5} "
            f"score={record['RRF_score']:.6f}"
        )

    return pd.DataFrame(records)


def reciprocal_rank(rank):
    if rank is None or pd.isna(rank):
        return 0.0
    rank = int(rank)
    return 1.0 / rank if rank > 0 else 0.0


def build_summary(results_df):
    """
    สรุป Accuracy / Hit@K / MRR
    เพื่อเปรียบเทียบ retrieval performance
    """
    summary = []

    methods = ["TFIDF", "BM25", "DENSE", "RRF"]

    for method in methods:
        ranks = results_df[f"{method}_rank"].astype(int)

        summary.append({
            "method": method,
            "Hit@1": float((ranks <= 1).mean()),
            "Hit@3": float((ranks <= 3).mean()),
            "Hit@5": float((ranks <= 5).mean()),
            "Hit@10": float((ranks <= 10).mean()),
            "MRR": float(
                np.mean([reciprocal_rank(r) for r in ranks])
            ),
            "MeanRank": float(ranks.mean()),
            "MedianRank": float(ranks.median()),
        })

    return pd.DataFrame(summary)


# ============================================================
# 8. DISPLAY RESULT
# ============================================================

def print_compact_result(results_df, summary_df):
    print("\n" + "=" * 110)
    print("SCORE / RANK: BM25 vs DENSE vs RRF")
    print("=" * 110)

    columns = [
        "query_no",
        "query_paraphrase",
        "BM25_score",
        "BM25_rank",
        "DENSE_score",
        "DENSE_rank",
        "RRF_score",
        "RRF_rank",
    ]

    display_df = results_df[columns].copy()

    display_df["BM25_score"] = display_df["BM25_score"].map(
        lambda x: f"{x:.4f}"
    )
    display_df["DENSE_score"] = display_df["DENSE_score"].map(
        lambda x: f"{x:.4f}"
    )
    display_df["RRF_score"] = display_df["RRF_score"].map(
        lambda x: f"{x:.6f}"
    )

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 220,
        "display.max_colwidth", 55,
    ):
        print(display_df.to_string(index=False))

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    summary_print = summary_df.copy()

    for col in ["Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR"]:
        summary_print[col] = summary_print[col].map(
            lambda x: f"{x:.4f}"
        )

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 160,
    ):
        print(summary_print.to_string(index=False))


# ============================================================
# 9. OPTIONAL INTERACTIVE SEARCH
# ============================================================

def interactive_search(
    corpus,
    bm25,
    tfidf,
    tfidf_matrix,
    dense_model,
    dense_embeddings,
    top_k=5,
):
    print("\n" + "=" * 90)
    print("INTERACTIVE HYBRID SEARCH")
    print("พิมพ์คำถามภาษาไทยได้เลย หรือพิมพ์ exit เพื่อออก")
    print("=" * 90)

    while True:
        try:
            query = input("\nQuery > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() in {"exit", "quit", "q"}:
            break

        if not query:
            continue

        result = search_all_methods(
            query=query,
            bm25=bm25,
            tfidf=tfidf,
            tfidf_matrix=tfidf_matrix,
            dense_model=dense_model,
            dense_embeddings=dense_embeddings,
        )

        print("\n[BM25]")
        for rank, idx in enumerate(
            result["bm25_ranking"][:top_k],
            start=1,
        ):
            idx = int(idx)
            print(
                f"{rank}. score={result['bm25_scores'][idx]:.4f}\n"
                f"   Q: {safe_text(corpus[idx].get('instruction'))}\n"
                f"   A: {safe_text(corpus[idx].get('answer'))}"
            )

        print("\n[DENSE]")
        for rank, idx in enumerate(
            result["dense_ranking"][:top_k],
            start=1,
        ):
            idx = int(idx)
            print(
                f"{rank}. score={result['dense_scores'][idx]:.4f}\n"
                f"   Q: {safe_text(corpus[idx].get('instruction'))}\n"
                f"   A: {safe_text(corpus[idx].get('answer'))}"
            )

        print("\n[HYBRID RRF]")
        for rank, idx in enumerate(
            result["rrf_ranking"][:top_k],
            start=1,
        ):
            idx = int(idx)
            print(
                f"{rank}. score={result['rrf_scores'][idx]:.6f}\n"
                f"   Q: {safe_text(corpus[idx].get('instruction'))}\n"
                f"   A: {safe_text(corpus[idx].get('answer'))}"
            )


# ============================================================
# 10. MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Thai Information Retrieval: "
            "TF-IDF + BM25 + Dense + RRF"
        )
    )

    parser.add_argument(
        "--json",
        default=DEFAULT_JSON,
        help=f"JSON corpus path (default: {DEFAULT_JSON})",
    )

    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help=f"Paraphrase CSV path (default: {DEFAULT_CSV})",
    )

    parser.add_argument(
        "--model",
        default=DENSE_MODEL_NAME,
        help=f"Dense model (default: {DENSE_MODEL_NAME})",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Embedding batch size (default: {BATCH_SIZE})",
    )

    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="จบโปรแกรมหลัง evaluation ไม่เข้า interactive search",
    )

    return parser.parse_args()


def main():
    # Windows console ให้แสดงภาษาไทยได้ดีขึ้น
    if os.name == "nt":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()

    json_path = Path(args.json)
    csv_path = Path(args.csv)

    if not json_path.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ JSON: {json_path.resolve()}"
        )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ CSV: {csv_path.resolve()}"
        )

    corpus, documents, queries = load_data(
        json_path=json_path,
        csv_path=csv_path,
    )

    gt_lookup = build_ground_truth_lookup(corpus)

    tokenized_docs, bm25, tfidf, tfidf_matrix = (
        build_lexical_models(documents)
    )

    dense_model, dense_embeddings = (
        build_dense_model_and_embeddings(
            documents=documents,
            model_name=args.model,
            cache_path=EMBEDDING_CACHE,
            batch_size=args.batch_size,
        )
    )

    results_df = evaluate(
        corpus=corpus,
        queries=queries,
        gt_lookup=gt_lookup,
        bm25=bm25,
        tfidf=tfidf,
        tfidf_matrix=tfidf_matrix,
        dense_model=dense_model,
        dense_embeddings=dense_embeddings,
    )

    print("\n[6/7] Building summary...")
    summary_df = build_summary(results_df)

    print_compact_result(
        results_df=results_df,
        summary_df=summary_df,
    )

    print("\n[7/7] Saving CSV outputs...")

    results_df.to_csv(
        DETAIL_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"  Saved -> {DETAIL_OUTPUT}")
    print(f"  Saved -> {SUMMARY_OUTPUT}")

    if not args.no_interactive:
        interactive_search(
            corpus=corpus,
            bm25=bm25,
            tfidf=tfidf,
            tfidf_matrix=tfidf_matrix,
            dense_model=dense_model,
            dense_embeddings=dense_embeddings,
            top_k=5,
        )


if __name__ == "__main__":
    main()
