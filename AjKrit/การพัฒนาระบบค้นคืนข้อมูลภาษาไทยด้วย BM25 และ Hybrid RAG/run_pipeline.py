from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


RESULT_COLUMNS = [
    "bm25_index_level",
    "bm25_score",
    "dense_index_level",
    "dense_score",
    "hybrid_index_level",
    "hybrid_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ค้นคืน FAQ ภาษาไทยด้วย TF-IDF, BM25, Dense และ Hybrid RRF "
            "แล้วเพิ่มผล Top-1 จำนวน 6 คอลัมน์ลง CSV"
        )
    )
    parser.add_argument("--json", type=Path, default=Path("data/thai_qa_utf8.json"))
    parser.add_argument("--csv", type=Path, default=Path("data/thai_qa_paraphrase_15.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/search_results.csv"))
    parser.add_argument(
        "--document-field",
        choices=["instruction", "input", "instruction+input"],
        default="instruction",
        help="ข้อความจาก JSON ที่ใช้สร้างคลังค้นหา (ค่าแนะนำสำหรับไฟล์นี้คือ instruction)",
    )
    parser.add_argument(
        "--query-column",
        default="instruction",
        help="ชื่อคอลัมน์คำค้นใน CSV",
    )
    parser.add_argument(
        "--index-field",
        default="__index_level_0__",
        help="ชื่อฟิลด์รหัสเอกสารใน JSON และ Ground Truth ใน CSV",
    )
    parser.add_argument(
        "--model",
        default="intfloat/multilingual-e5-small",
        help="SentenceTransformer model สำหรับ Dense Search",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="บังคับสร้าง Dense document embeddings ใหม่",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="ตรวจโครงสร้างไฟล์โดยไม่โหลดโมเดลหรือคำนวณคะแนน",
    )
    return parser.parse_args()


def thai_tokens(text: object) -> list[str]:
    from pythainlp.tokenize import word_tokenize

    return [
        token.strip()
        for token in word_tokenize(str(text), engine="newmm", keep_whitespace=False)
        if token.strip()
    ]


def read_inputs(args: argparse.Namespace) -> tuple[list[dict], pd.DataFrame]:
    if not args.json.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ JSON: {args.json}")
    if not args.csv.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ CSV: {args.csv}")

    with args.json.open("r", encoding="utf-8-sig") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError("JSON ต้องเป็น list ที่มีอย่างน้อย 1 record")

    csv_df = pd.read_csv(args.csv, encoding="utf-8-sig")
    if csv_df.empty:
        raise ValueError("CSV ไม่มีแถวข้อมูล")
    if args.query_column not in csv_df.columns:
        raise ValueError(f"CSV ไม่มีคอลัมน์คำค้น '{args.query_column}'")
    if csv_df[args.query_column].isna().any():
        bad_rows = (csv_df.index[csv_df[args.query_column].isna()] + 2).tolist()
        raise ValueError(f"คอลัมน์คำค้นมีค่าว่างที่บรรทัด: {bad_rows}")

    required_json_fields = {args.index_field}
    if args.document_field == "instruction+input":
        required_json_fields.update({"instruction", "input"})
    else:
        required_json_fields.add(args.document_field)

    missing_by_field = {
        field: [i for i, row in enumerate(records) if field not in row]
        for field in required_json_fields
    }
    missing_by_field = {k: v for k, v in missing_by_field.items() if v}
    if missing_by_field:
        summary = ", ".join(f"{k}: {v[:5]}" for k, v in missing_by_field.items())
        raise ValueError(f"JSON มี record ที่ขาดฟิลด์จำเป็น ({summary})")

    return records, csv_df


def build_documents(records: list[dict], document_field: str) -> list[str]:
    if document_field == "instruction+input":
        return [f"{row['instruction']} {row['input']}".strip() for row in records]
    return [str(row[document_field]).strip() for row in records]


def rank_desc(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return order and 1-based rank for each corpus position."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return order, ranks


def normalize_index_value(value: object) -> str | None:
    """Make integer-looking CSV/JSON IDs comparable even when pandas uses float."""
    if pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def corpus_fingerprint(documents: Iterable[str], model_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    for text in documents:
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()[:20]


def load_or_encode_documents(
    model,
    documents: list[str],
    cache_dir: Path,
    model_name: str,
    batch_size: int,
    rebuild_cache: bool,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = corpus_fingerprint(documents, model_name)
    cache_file = cache_dir / f"dense_documents_{fingerprint}.npy"

    if cache_file.exists() and not rebuild_cache:
        print(f"ใช้ Dense cache: {cache_file}")
        embeddings = np.load(cache_file)
        if embeddings.shape[0] == len(documents):
            return embeddings
        print("จำนวน embeddings ใน cache ไม่ตรงกับเอกสาร จึงสร้างใหม่")

    print(f"กำลังสร้าง Dense embeddings สำหรับ {len(documents):,} เอกสาร...")
    embeddings = model.encode(
        ["passage: " + text for text in documents],
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(cache_file, embeddings)
    print(f"บันทึก Dense cache: {cache_file}")
    return embeddings


def evaluate_top1(
    csv_df: pd.DataFrame,
    index_field: str,
    result_columns: list[str],
) -> None:
    if index_field not in csv_df.columns:
        print(f"ไม่พบ Ground Truth '{index_field}' ใน CSV จึงข้าม Hit@1")
        return

    truth = csv_df[index_field].map(normalize_index_value)
    valid = truth.notna() & truth.ne("")
    if not valid.any():
        print(f"Ground Truth '{index_field}' ว่างทั้งหมด จึงข้าม Hit@1")
        return
    print("\nผล Hit@1 เทียบกับ Ground Truth")
    for column in result_columns:
        predictions = csv_df[column].map(normalize_index_value)
        hits = predictions[valid].eq(truth[valid])
        print(f"- {column}: {hits.mean():.4f} ({hits.sum()}/{len(hits)})")


def main() -> int:
    args = parse_args()
    records, csv_df = read_inputs(args)
    documents = build_documents(records, args.document_field)
    queries = csv_df[args.query_column].astype(str).str.strip().tolist()
    index_values = np.array(
        [normalize_index_value(row[args.index_field]) for row in records], dtype=object
    )

    print(f"JSON records: {len(records):,}")
    print(f"CSV queries: {len(queries):,}")
    print(f"Document field: {args.document_field}")
    print(f"Query column: {args.query_column}")
    duplicate_count = len(index_values) - len(set(index_values.tolist()))
    if duplicate_count:
        print(
            f"คำเตือน: JSON มีค่า {args.index_field} ซ้ำ {duplicate_count:,} records; "
            "ผลยังส่งออกตามฟิลด์นี้ได้ แต่ ID ไม่ใช่รหัสเฉพาะทั้งไฟล์"
        )
    if args.validate_only:
        print("ตรวจโครงสร้างไฟล์ผ่าน")
        return 0

    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError(
            "ยังไม่ได้ติดตั้ง rank-bm25: "
            "รัน 'python -m pip install -r requirements.txt'"
        ) from exc

    print("กำลังตัดคำภาษาไทยและสร้าง BM25...")
    tokenized_documents = [thai_tokens(text) for text in documents]
    bm25 = BM25Okapi(tokenized_documents)

    print("กำลังสร้าง TF-IDF baseline...")
    tfidf = TfidfVectorizer(
        tokenizer=thai_tokens,
        token_pattern=None,
        lowercase=False,
        dtype=np.float32,
    )
    tfidf_documents = tfidf.fit_transform(documents)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "ยังไม่ได้ติดตั้ง sentence-transformers: "
            "รัน 'python -m pip install -r requirements.txt'"
        ) from exc

    print(f"กำลังโหลด Dense model: {args.model}")
    dense_model = SentenceTransformer(args.model)
    dense_documents = load_or_encode_documents(
        model=dense_model,
        documents=documents,
        cache_dir=args.cache_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        rebuild_cache=args.rebuild_cache,
    )
    dense_queries = dense_model.encode(
        ["query: " + query for query in queries],
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    output_rows: list[dict] = []
    tfidf_top1_ids: list[str] = []
    print("กำลังจัดอันดับ BM25, Dense และ Hybrid RRF...")
    for query_number, (query, dense_query) in enumerate(
        zip(queries, dense_queries), start=1
    ):
        bm25_scores = np.asarray(bm25.get_scores(thai_tokens(query)), dtype=np.float64)
        dense_scores = dense_documents @ dense_query

        bm25_order, bm25_ranks = rank_desc(bm25_scores)
        dense_order, dense_ranks = rank_desc(dense_scores)
        hybrid_scores = (
            1.0 / (args.rrf_k + bm25_ranks)
            + 1.0 / (args.rrf_k + dense_ranks)
        )
        hybrid_order, _ = rank_desc(hybrid_scores)

        tfidf_query = tfidf.transform([query])
        tfidf_scores = (tfidf_query @ tfidf_documents.T).toarray()[0]
        tfidf_top1_ids.append(str(index_values[int(np.argmax(tfidf_scores))]))

        bm25_pos = int(bm25_order[0])
        dense_pos = int(dense_order[0])
        hybrid_pos = int(hybrid_order[0])
        output_rows.append(
            {
                "bm25_index_level": index_values[bm25_pos],
                "bm25_score": float(bm25_scores[bm25_pos]),
                "dense_index_level": index_values[dense_pos],
                "dense_score": float(dense_scores[dense_pos]),
                "hybrid_index_level": index_values[hybrid_pos],
                "hybrid_score": float(hybrid_scores[hybrid_pos]),
            }
        )
        print(f"  เสร็จแล้ว {query_number}/{len(queries)}")

    for column in RESULT_COLUMNS:
        csv_df[column] = [row[column] for row in output_rows]

    for score_column in ["bm25_score", "dense_score", "hybrid_score"]:
        csv_df[score_column] = csv_df[score_column].round(8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nเขียนผลลัพธ์แล้ว: {args.output}")
    print("คอลัมน์ใหม่:", ", ".join(RESULT_COLUMNS))

    evaluate_top1(
        csv_df,
        args.index_field,
        ["bm25_index_level", "dense_index_level", "hybrid_index_level"],
    )
    if args.index_field in csv_df.columns:
        truth = csv_df[args.index_field].map(normalize_index_value).reset_index(drop=True)
        valid = truth.notna() & truth.ne("")
        if valid.any():
            predictions = pd.Series(tfidf_top1_ids).map(normalize_index_value)
            tfidf_hits = predictions[valid].eq(truth[valid])
            print(
                f"- tfidf_index_level (console only): {tfidf_hits.mean():.4f} "
                f"({tfidf_hits.sum()}/{len(tfidf_hits)})"
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
