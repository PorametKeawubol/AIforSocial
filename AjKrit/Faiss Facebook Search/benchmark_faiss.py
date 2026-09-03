#!/usr/bin/env python3
"""Compare three Thai text encoders with FAISS cosine and L2 search.

The experiment uses unit-normalized vectors.  On the unit hypersphere the
following identities must hold (up to floating-point error):

    cosine_similarity = 1 - squared_l2_distance / 2
    l2_distance = sqrt(2 * (1 - cosine_similarity))

Run from this directory, for example:
    python benchmark_faiss.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


CORPUS = [
    ["สำนักงานใหญ่ของคุณตั้งอยู่ที่ไหน?", "location"],
    ["โยนโทรศัพท์ลงน้ำ", "random"],
    ["การควบคุมการเข้าถึงเครือข่าย", "networking"],
    ["ที่อยู่บริษัท", "location"],
    ["ติดต่อฝ่ายไอทีได้อย่างไร", "support"],
]

MODELS = {
    "WangchanBERTa": "kornwtp/ConGen-model-wangchanberta",
    "PhayaThaiBERT": "kornwtp/SCT-model-phayathaibert",
    "Multilingual-MPNet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}

DEFAULT_QUERY = "ที่ทำการออฟฟิศอยู่ที่ไหน?"
BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Thai sentence encoders with FAISS IndexFlatIP and IndexFlatL2."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Thai text to search for.")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="SentenceTransformer device; default chooses CUDA when available.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=BASE_DIR / "faiss_model_comparison_results.csv",
        help="Where to save the results table.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=BASE_DIR / "faiss_model_comparison_results.json",
        help="Where to save full-precision results and experiment metadata.",
    )
    return parser.parse_args()


def choose_device(requested: str | None) -> str:
    """Use the requested device, otherwise pick CUDA only when it is available."""
    if requested:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def as_faiss_vectors(vectors: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 matrix, the format FAISS expects."""
    return np.ascontiguousarray(vectors, dtype=np.float32)


def benchmark_model(
    model_name: str,
    model_path: str,
    texts: list[str],
    categories: list[str],
    query: str,
    device: str,
) -> dict[str, object]:
    """Run exact top-1 search with both FAISS metrics for one encoder."""
    print(f"\nLoading {model_name}: {model_path} ({device})", flush=True)
    started = time.perf_counter()
    encoder = SentenceTransformer(model_path, device=device)

    corpus_vectors = as_faiss_vectors(
        encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    )
    faiss.normalize_L2(corpus_vectors)
    dimension = int(corpus_vectors.shape[1])

    # Inner product of two L2-normalized vectors equals cosine similarity.
    index_ip = faiss.IndexFlatIP(dimension)
    index_ip.add(corpus_vectors)

    # IndexFlatL2 returns *squared* Euclidean distances.
    index_l2 = faiss.IndexFlatL2(dimension)
    index_l2.add(corpus_vectors)

    query_vector = as_faiss_vectors(
        encoder.encode([query], convert_to_numpy=True, show_progress_bar=False)
    )
    faiss.normalize_L2(query_vector)

    cosine_scores, cosine_indices = index_ip.search(query_vector, k=1)
    l2_squared_scores, l2_indices = index_l2.search(query_vector, k=1)
    elapsed_seconds = time.perf_counter() - started

    cosine_index = int(cosine_indices[0, 0])
    l2_index = int(l2_indices[0, 0])
    actual_cosine = float(cosine_scores[0, 0])
    actual_l2 = float(np.sqrt(max(float(l2_squared_scores[0, 0]), 0.0)))

    # Mathematical verification on normalized vectors.
    calculated_cosine = 1.0 - (actual_l2**2) / 2.0
    calculated_l2 = float(np.sqrt(max(2.0 * (1.0 - actual_cosine), 0.0)))

    return {
        "Model": model_name,
        "Model Path": model_path,
        "Vector Dimension": dimension,
        "Matched Text": texts[cosine_index],
        "Matched Category": categories[cosine_index],
        "IP Index": cosine_index,
        "L2 Index": l2_index,
        "Indexes Agree": cosine_index == l2_index,
        "Cosine (Actual)": actual_cosine,
        "L2 Dist (Actual)": actual_l2,
        "Cosine (Calc from L2)": calculated_cosine,
        "L2 Dist (Calc from Cos)": calculated_l2,
        "Cosine Absolute Error": abs(actual_cosine - calculated_cosine),
        "L2 Absolute Error": abs(actual_l2 - calculated_l2),
        "Elapsed Seconds": elapsed_seconds,
    }


def display_table(results: list[dict[str, object]]) -> None:
    """Print the lab's result table rounded only for display."""
    table = pd.DataFrame(results)[
        [
            "Model",
            "Matched Text",
            "Matched Category",
            "Cosine (Actual)",
            "L2 Dist (Actual)",
            "Cosine (Calc from L2)",
            "L2 Dist (Calc from Cos)",
            "Cosine Absolute Error",
            "L2 Absolute Error",
        ]
    ].copy()
    numeric_columns = table.select_dtypes(include="number").columns
    table[numeric_columns] = table[numeric_columns].round(6)
    print("\n=== Results ===")
    print(table.to_string(index=False))


def save_results(
    results: list[dict[str, object]], query: str, device: str, csv_path: Path, json_path: Path
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(
            {
                "query": query,
                "device": device,
                "normalization": "L2 normalization applied to corpus and query vectors",
                "formula": "cosine = 1 - squared_l2_distance / 2",
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved CSV : {csv_path}")
    print(f"Saved JSON: {json_path}")


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    frame = pd.DataFrame(CORPUS, columns=["text", "category"])
    texts = frame["text"].tolist()
    categories = frame["category"].tolist()

    print(f"=== Search query: {args.query!r} ===")
    print("Using L2-normalized vectors for both FAISS indexes.")

    results = [
        benchmark_model(name, path, texts, categories, args.query, device)
        for name, path in MODELS.items()
    ]
    display_table(results)
    save_results(results, args.query, device, args.output_csv, args.output_json)

    if not all(result["Indexes Agree"] for result in results):
        print("Warning: IP and L2 selected different top-1 items; inspect the full results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
