"""Reproducible model comparison, isolated from the production index and .env.

Embedding evaluation uses section-level weak labels; it does not pretend all
chunks in a section have been manually annotated as relevant. LLM evaluation
uses frozen production contexts and one common answer scorer across all models.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import platform
import importlib.metadata
import os
import time
from pathlib import Path

import faiss
import httpx
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import ROOT, settings
from evaluate import answer_body, cosine, read_dataset, semantic_faithfulness
from llm import FALLBACK, OllamaClient, citation_sources, context_results
from models import DocumentChunk, RetrievalResult
from retriever import HybridRetriever, expand_thai_query, reciprocal_rank_fusion, thai_tokens

EMBED_MODELS = ["BAAI/bge-m3", "intfloat/multilingual-e5-large", "Qwen/Qwen3-Embedding-0.6B"]
LLM_MODELS = ["ministral-3:8b", "qwen3.5:9b", "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"]


def encode(encoder, model: str, texts: list[str], query: bool, **kwargs):
    if "multilingual-e5" in model:
        texts = [("query: " if query else "passage: ") + text for text in texts]
        kwargs["prompt"] = ""
    elif "Qwen3-Embedding" in model:
        kwargs["prompt"] = ("Instruct: Given a question in Thai, retrieve relevant passages from the Thai police citizen manual that answer the question.\nQuery: " if query else "")
    else:
        kwargs["prompt"] = ""
    return encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True, **kwargs).astype("float32")


def section_metrics(sections: list[str], target: str, k: int) -> dict:
    ranked = list(dict.fromkeys(sections))[:k]
    hit = int(target in ranked)
    precision = hit / len(ranked) if ranked else 0.0
    recall = float(hit)  # one annotated relevant section per question
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "mrr": 1 / (ranked.index(target) + 1) if hit else 0.,
            "hit_at_1": float(bool(ranked) and ranked[0] == target), "sections": ranked}


def latency_stats(values: list[float]) -> dict:
    return {"latency_mean_s": float(np.mean(values)), "latency_p50_s": float(np.percentile(values, 50)),
            "latency_p95_s": float(np.percentile(values, 95)), "latency_samples": len(values)} if values else {}


def write_report(path: Path, report: dict):
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def embedding_benchmark(args, dataset, chunks, report):
    bm25 = BM25Okapi([thai_tokens(c.text) for c in chunks])
    for model in args.embedding_models:
        if any(r.get("model") == model for r in report["embeddings"]):
            continue
        print(f"Embedding model: {model}", flush=True)
        started = time.perf_counter()
        encoder = SentenceTransformer(model, device="cpu")
        encoder.float()  # Decoder embedding checkpoints may default to slow CPU BF16.
        encoder.max_seq_length = args.max_seq_length
        loaded = time.perf_counter() - started
        started = time.perf_counter()
        vectors = encode(encoder, model, [c.text for c in chunks], False, batch_size=args.batch_size, show_progress_bar=True)
        index = faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors)
        build_seconds = time.perf_counter() - started
        model_dir = args.out / model.replace("/", "--")
        model_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(model_dir / "index.faiss"))
        write_report(model_dir / "manifest.json", {"embedding_model": model, "dimension": vectors.shape[1],
                     "max_seq_length": args.max_seq_length, "chunks_sha256": report["protocol"]["chunks_sha256"],
                     "note": "Benchmark index only; use the same source chunks and encode() query prompt."})
        warm_started = time.perf_counter()
        encode(encoder, model, [expand_thai_query(dataset[0]["question"])], True)
        warmup_s = time.perf_counter() - warm_started
        rows = []; dense_times = []; hybrid_times = []
        for item in dataset:
            question = expand_thai_query(item["question"])
            timings = []; hybrid_timings = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                query = encode(encoder, model, [question], True)
                scores, ids = index.search(query, len(chunks))
                dense_s = time.perf_counter() - started
                timings.append(dense_s)
                sparse = np.asarray(bm25.get_scores(thai_tokens(question)))
                sparse_ids = np.argsort(-sparse, kind="stable")[:args.candidate_k].tolist()
                dense_ids = ids[0][:args.candidate_k].tolist()
                fused = reciprocal_rank_fusion({i: r for r, i in enumerate(dense_ids, 1)},
                                               {i: r for r, i in enumerate(sparse_ids, 1)})
                hybrid_ids = sorted(fused, key=lambda i: (-fused[i], i))
                hybrid_timings.append(time.perf_counter() - started)
            target = item["target_section"]
            positives = [float(s) for s, i in zip(scores[0], ids[0]) if chunks[int(i)].section_id == target]
            rows.append({"id": item["id"], "question": item["question"], "target_section": target,
                         "dense": section_metrics([chunks[int(i)].section_id for i in ids[0]], target, args.top_k),
                         "hybrid_rrf": section_metrics([chunks[i].section_id for i in hybrid_ids], target, args.top_k),
                         "cosine_best_target": max(positives) if positives else None,
                         "cosine_top1": float(scores[0][0]), "dense_seconds": timings, "hybrid_seconds": hybrid_timings})
            dense_times.extend(timings); hybrid_times.extend(hybrid_timings)
            print(f"  {item['id']} dense={rows[-1]['dense']['recall']:.0f} hybrid={rows[-1]['hybrid_rrf']['recall']:.0f} {np.mean(timings):.3f}s", flush=True)
        for method, times in (("dense", dense_times), ("hybrid_rrf", hybrid_times)):
            summary = {"model": model, "method": method, "dtype": str(next(encoder.parameters()).dtype), "dimension": vectors.shape[1], "queries": len(rows),
                       "load_download_seconds": loaded, "build_seconds": build_seconds, "query_warmup_seconds": warmup_s,
                       **{key + f"_at_{args.top_k}": float(np.mean([r[method][key] for r in rows])) for key in ("precision", "recall", "f1", "mrr")},
                       "hit_at_1": float(np.mean([r[method]["hit_at_1"] for r in rows])),
                       "cosine_best_target_mean": float(np.mean([r["cosine_best_target"] for r in rows if r["cosine_best_target"] is not None])),
                       **latency_stats(times)}
            report["embeddings"].append(summary)
        write_report(model_dir / "rows.json", {"model": model, "rows": rows})
        write_report(args.out / "comparison.json", report)
        del encoder, index, vectors; gc.collect()


def freeze_contexts(args, dataset, report):
    path = args.out / "frozen_contexts.json"
    if path.exists():
        data = json.loads(path.read_text())
        if data["dataset_sha256"] != report["protocol"]["dataset_sha256"] or data["chunks_sha256"] != report["protocol"]["chunks_sha256"]:
            raise ValueError("Frozen contexts do not match inputs; choose a fresh output directory")
        if [r["id"] for r in data["rows"]] != [r["id"] for r in dataset]:
            raise ValueError("Frozen context IDs/order do not match dataset")
        return data["rows"]
    retriever = HybridRetriever.load(settings.index_dir)
    rows = []
    for item in dataset:
        results = retriever.search(item["question"], top_k=args.top_k, candidate_k=args.candidate_k)
        rows.append({"id": item["id"], "results": [r.to_dict() for r in results]})
        print(f"Frozen context: {item['id']} ({len(rows)}/{len(dataset)})", flush=True)
    write_report(path, {"dataset_sha256": report["protocol"]["dataset_sha256"], "chunks_sha256": report["protocol"]["chunks_sha256"],
                      "embedding_model": retriever.embedding_model, "rerank_model": retriever.rerank_model, "rows": rows})
    del retriever; gc.collect()
    return rows


def restore_results(row):
    return [RetrievalResult(chunk=DocumentChunk.from_dict(r["chunk"]),
                           **{k: r[k] for k in ("dense_score", "sparse_score", "dense_rank", "sparse_rank", "rrf_score", "weighted_score", "rerank_score") if k in r}) for r in row["results"]]


def token_f1(answer, reference):
    from collections import Counter
    a, b = Counter(thai_tokens(answer)), Counter(thai_tokens(reference))
    overlap = sum((a & b).values())
    p = overlap / sum(a.values()) if a else 0.
    r = overlap / sum(b.values()) if b else 0.
    return p, r, 2 * p * r / (p + r) if p + r else 0.


def llm_benchmark(args, dataset, report):
    frozen = freeze_contexts(args, dataset, report)
    with httpx.Client(timeout=20) as client:
        response = client.get(f"{settings.ollama_host}/api/tags"); response.raise_for_status()
        installed = {m["name"]: m.get("digest") for m in response.json()["models"]}
        metadata = {}
        for model in args.llm_models:
            if model in installed:
                info = client.post(f"{settings.ollama_host}/api/show", json={"model": model})
                info.raise_for_status(); info = info.json()
                metadata[model] = {"digest": installed[model], "details": info.get("details"),
                                   "architecture": info.get("model_info", {}).get("general.architecture"),
                                   "parameter_count": info.get("model_info", {}).get("general.parameter_count")}
        report.setdefault("ollama_models", {}).update(metadata)
    for model in args.llm_models:
        if any(r["model"] == model for r in report["llms"]):
            continue
        if model not in installed:
            raise ValueError(f"Missing Ollama model: {model}; run ollama pull {model}")
        path = args.out / (model.replace("/", "--").replace(":", "_") + "_answers.json")
        checkpoint = json.loads(path.read_text()) if path.exists() else {"model": model, "digest": installed[model], "rows": []}
        if checkpoint["digest"] != installed[model]:
            raise ValueError("Ollama model digest changed; choose a fresh output directory")
        rows = checkpoint["rows"]
        llm = OllamaClient(settings.ollama_host, model, timeout_seconds=args.timeout, max_answer_tokens=args.max_answer_tokens)
        print(f"LLM: {model} ({len(rows)}/{len(dataset)} checkpoint)", flush=True)
        started = time.perf_counter()
        _, warm_metrics = llm.chat([{"role": "user", "content": 'ตอบ JSON สั้น ๆ: {"ready":true}'}])
        checkpoint["warmup_seconds"] = time.perf_counter() - started
        checkpoint["warmup_metrics"] = warm_metrics
        for item, context in zip(dataset[len(rows):], frozen[len(rows):]):
            results = restore_results(context)
            started = time.perf_counter()
            try:
                result = llm.answer_from_context(item["question"], results, max_context_chars=args.max_context_chars)
                valid = result.answer != FALLBACK and bool(result.citations)
                sources = citation_sources(results)
                evidence = "\n".join(r.chunk.parent_text or r.chunk.text for s, r in zip(sources, context_results(results)) if s["id"] in result.citations)
                p, recall, f1 = token_f1(answer_body(result.answer), item["reference_answer"])
                row = {"id": item["id"], "question": item["question"], "answer": result.answer,
                       "reference": item["reference_answer"], "evidence": evidence, "citations": result.citations,
                       "grounded_citation_contract": valid, "token_precision": p, "token_recall": recall, "token_f1": f1,
                       "metrics": result.metrics, "error": None}
            except Exception as exc:
                row = {"id": item["id"], "question": item["question"], "answer": "", "reference": item["reference_answer"],
                       "evidence": "", "citations": [], "grounded_citation_contract": False,
                       "token_precision": 0., "token_recall": 0., "token_f1": 0., "metrics": {},
                       "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            row["latency_seconds"] = time.perf_counter() - started
            rows.append(row); write_report(path, checkpoint)
            print(f"  {item['id']} {row['latency_seconds']:.2f}s citation={row['grounded_citation_contract']} error={row['error']}", flush=True)
        # Fixed multilingual encoder for all answer models, never their own embeddings.
        scorer = SentenceTransformer("BAAI/bge-m3", device="cpu"); scorer.float(); scorer.max_seq_length = 512
        av = scorer.encode([answer_body(r["answer"]) for r in rows], normalize_embeddings=True, convert_to_numpy=True)
        rv = scorer.encode([r["reference"] for r in rows], normalize_embeddings=True, convert_to_numpy=True)
        qv = scorer.encode([r["question"] for r in rows], normalize_embeddings=True, convert_to_numpy=True)
        for row, a, r, q in zip(rows, av, rv, qv):
            row["reference_cosine"] = cosine(a, r) if not row["error"] else 0.
            row["answer_relevance_cosine"] = cosine(a, q) if not row["error"] else 0.
            row["semantic_faithfulness_proxy"] = semantic_faithfulness(row["answer"], row["evidence"], scorer) if row["grounded_citation_contract"] else 0.
        del scorer; gc.collect()
        if not args.no_bert_score:
            from bert_score import score
            p, r, f = score([row["answer"] or "ไม่มีคำตอบ" for row in rows], [row["reference"] for row in rows],
                            lang="th", model_type="bert-base-multilingual-cased", device="cpu", batch_size=4, verbose=False)
            for row, pv, rv, fv in zip(rows, p.tolist(), r.tolist(), f.tolist()):
                row.update({"bertscore_precision": pv if not row["error"] else 0., "bertscore_recall": rv if not row["error"] else 0., "bertscore_f1": fv if not row["error"] else 0.})
        write_report(path, checkpoint)
        keys = ("token_precision", "token_recall", "token_f1", "reference_cosine", "answer_relevance_cosine", "semantic_faithfulness_proxy", "bertscore_precision", "bertscore_recall", "bertscore_f1")
        summary = {"model": model, "digest": installed[model], "queries": len(rows), "errors": sum(bool(r["error"]) for r in rows),
                   "grounded_citation_contract_rate": float(np.mean([r["grounded_citation_contract"] for r in rows])),
                   "warmup_seconds": checkpoint["warmup_seconds"],
                   **{key: float(np.mean([r[key] for r in rows])) for key in keys if all(key in r for r in rows)},
                   **latency_stats([r["latency_seconds"] for r in rows])}
        tps = [r["metrics"]["eval_count"] / (r["metrics"]["eval_duration_ns"] / 1e9) for r in rows if r["metrics"].get("eval_duration_ns") and r["metrics"].get("eval_count")]
        summary["generation_tokens_per_second_mean"] = float(np.mean(tps)) if tps else None
        report["llms"].append(summary); write_report(args.out / "comparison.json", report)
        export_tables(args.out, report)


def archive_llm(report: dict, model: str):
    """Remove a comparator from the main table while retaining historical scores."""
    removed = [r for r in report["llms"] if r["model"] == model]
    existing = {r["model"] for r in report.get("archived_llms", [])}
    report.setdefault("archived_llms", []).extend(r for r in removed if r["model"] not in existing)
    report["llms"] = [r for r in report["llms"] if r["model"] != model]


def export_tables(out, report):
    specs = [("embeddings", ["model", "method", "hit_at_1", "recall_at_5", "precision_at_5", "f1_at_5", "mrr_at_5", "cosine_best_target_mean", "latency_p95_s"]),
             ("llms", ["model", "grounded_citation_contract_rate", "bertscore_precision", "bertscore_recall", "bertscore_f1", "reference_cosine", "semantic_faithfulness_proxy", "latency_p95_s", "errors"])]
    lines = ["# Model benchmark", "", "Status: " + report.get("status", "running"), "",
             "Same dataset/chunks. Embedding: CPU FP32, 512-token budget, section-level weak labels; dense and hybrid RRF, no reranker.",
             "LLM: frozen production BGE-M3 contexts; identical JSON prompt, temperature=0, think=false, num_ctx=8192 and token budget.",
             "F1@k is section retrieval F1, NOT BERTScore F1. One target section: Recall@k equals Hit@k. At k=5, perfect recall typically gives precision=.2 and F1=.333.",
             "Cosine scales differ across embedding models; compare ranking metrics, not absolute cosine. Semantic faithfulness and citation-contract rate are proxies, not factual correctness.",
             "Latency excludes downloads/loading/warmup; LLM latency includes prompt processing + generation, not retrieval or LINE/network delivery. Single LLM pass: p95 based on 18 queries is exploratory.",
             "Live LINE shares Ollama: concurrent traffic invalidates isolated latency assumptions. No self-judge scores are compared across different judges.", ""]
    lines += ["The current dataset has 18 in-domain questions; it does not measure out-of-document refusal, chat-history correctness or broader model ability.",
              "Selected qualitative findings are in review_notes.md when present; similarity scores can miss negation and mixing procedures.", ""]
    for label, columns in specs:
        rows = report[label]
        if label == "embeddings":
            k = report["protocol"]["top_k"]
            columns = [c.replace("_at_5", f"_at_{k}") for c in columns]
        if not rows:
            continue
        with (out / f"{label}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
            writer.writeheader(); writer.writerows(rows)
        lines += [f"## {label}", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        for row in rows:
            lines.append("| " + " | ".join(f"{row[c]:.4f}" if isinstance(row.get(c), float) else str(row.get(c, "—")) for c in columns) + " |")
        lines.append("")
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("embeddings", "llms", "all"), default="all")
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval/questions.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/model_benchmark")
    parser.add_argument("--embedding-models", nargs="+", default=EMBED_MODELS)
    parser.add_argument("--llm-models", nargs="+", default=LLM_MODELS)
    parser.add_argument("--archive-llm", help="Move the old comparator's summary to archived_llms, preserving answer files")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-answer-tokens", type=int, default=420)
    parser.add_argument("--max-context-chars", type=int, default=10000)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--no-bert-score", action="store_true")
    args = parser.parse_args()
    if min(args.top_k, args.candidate_k, args.repeats, args.batch_size, args.threads, args.max_seq_length, args.max_answer_tokens, args.max_context_chars, args.timeout) <= 0:
        parser.error("All numeric parameters must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    lock_file = (args.out / ".benchmark.lock").open("a")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another benchmark is using this output directory") from None
    torch.set_num_threads(args.threads); faiss.omp_set_num_threads(args.threads)
    dataset = read_dataset(args.dataset)
    chunk_path = settings.index_dir / "chunks.json"
    chunks = [DocumentChunk.from_dict(c) for c in json.loads(chunk_path.read_text())]
    if any(not r.get("target_section") or not r.get("reference_answer") for r in dataset):
        raise ValueError("Every benchmark question requires target_section and reference_answer")
    if any(r["target_section"] not in {c.section_id for c in chunks} for r in dataset):
        raise ValueError("Dataset target_section does not exist in the source chunks")
    protocol = {"dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(), "chunks_sha256": hashlib.sha256(chunk_path.read_bytes()).hexdigest(),
                "production_manifest_sha256": hashlib.sha256((settings.index_dir / "manifest.json").read_bytes()).hexdigest(),
                "top_k": args.top_k, "candidate_k": args.candidate_k, "repeats": args.repeats, "threads": args.threads,
                "batch_size": args.batch_size, "max_seq_length": args.max_seq_length,
                "max_answer_tokens": args.max_answer_tokens, "max_context_chars": args.max_context_chars,
                "timeout": args.timeout, "bert_score": not args.no_bert_score, "python": platform.python_version(), "torch": torch.__version__}
    path = args.out / "comparison.json"
    report = json.loads(path.read_text()) if path.exists() else {"protocol": protocol, "embeddings": [], "llms": []}
    if report["protocol"] != protocol:
        raise ValueError("Benchmark protocol changed; use a new --out directory")
    if args.archive_llm:
        if args.stage == "embeddings" or args.archive_llm in args.llm_models:
            parser.error("Archive only an LLM outside the requested LLM comparison")
        archive_llm(report, args.archive_llm)
    cpu_lines = Path("/proc/cpuinfo").read_text().splitlines() if Path("/proc/cpuinfo").exists() else []
    report["environment"] = {"platform": platform.platform(), "logical_cpus": os.cpu_count(),
                             "cpu": next((line.split(":", 1)[1].strip() for line in cpu_lines if line.startswith("model name")), platform.processor()),
                             "packages": {name: importlib.metadata.version(name) for name in ("sentence-transformers", "transformers", "faiss-cpu", "bert-score")}}
    report["status"] = "running"; write_report(path, report)
    try:
        if args.stage in ("embeddings", "all"):
            embedding_benchmark(args, dataset, chunks, report)
        if args.stage in ("llms", "all"):
            llm_benchmark(args, dataset, report)
        report["status"] = "requested_stage_complete"
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "interrupted"; report["last_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_report(path, report); export_tables(args.out, report)


if __name__ == "__main__":
    main()
