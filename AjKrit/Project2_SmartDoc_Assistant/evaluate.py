"""Quantitative evaluation for retrieval and generated answers.

The default run evaluates retrieval and does not call Ollama.  Add
``--run-llm`` to generate answers with Ollama and measure SBERT similarity,
semantic faithfulness, answer relevance and (optionally) BERTScore.  The
optional ``--llm-judge`` adds a local LLM judge for faithfulness/relevance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from assistant import SmartDocAssistant
from config import settings
from llm import citation_sources, context_results
from retriever import thai_tokens


def read_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError("Evaluation dataset is empty")
    return rows


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def faithfulness_heuristic(answer: str, cited_text: str) -> float:
    """Token-overlap proxy; report as heuristic, not a human truth label."""

    answer_tokens = {token for token in thai_tokens(answer) if not token.startswith("s")}
    evidence_tokens = set(thai_tokens(cited_text))
    if not answer_tokens:
        return 0.0
    return min(1.0, len(answer_tokens & evidence_tokens) / len(answer_tokens))


def answer_body(answer: str) -> str:
    """Remove generated source footer before scoring answer semantics."""

    return re.split(r"\n\s*ที่มา\s*:\s*", answer, maxsplit=1)[0].strip()


def text_sentences(text: str) -> list[str]:
    """Split Thai/English answer or evidence into scoreable propositions."""

    pieces = re.split(r"\n+|(?<=[.!?！？ฯ])\s*", text)
    return [piece.strip() for piece in pieces if len(piece.strip()) >= 8]


def semantic_faithfulness(
    answer: str,
    cited_text: str,
    encoder: Any,
    threshold: float = 0.52,
) -> float:
    """Estimate the fraction of answer propositions supported by evidence.

    This is an embedding-based RAG metric, not a legal-truth oracle.  Each
    answer proposition is compared with the strongest evidence sentence and
    counted as supported when cosine similarity reaches ``threshold``.
    """

    answer_units = text_sentences(answer_body(answer))
    evidence_units = text_sentences(cited_text)
    if not answer_units or not evidence_units:
        return 0.0
    answer_vectors = encoder.encode(answer_units, normalize_embeddings=True, convert_to_numpy=True)
    evidence_vectors = encoder.encode(evidence_units, normalize_embeddings=True, convert_to_numpy=True)
    similarities = answer_vectors @ evidence_vectors.T
    best = similarities.max(axis=1)
    return float(np.mean(best >= threshold))


def mean_or_none(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 6) if values else None


def llm_judge(
    assistant: SmartDocAssistant,
    question: str,
    answer: str,
    cited_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ask the local LLM to score support and relevance using strict JSON."""

    system = (
        "คุณเป็นผู้ประเมินคำตอบ RAG ภาษาไทยแบบเข้มงวด\n"
        "ประเมินเฉพาะข้อมูลที่อยู่ใน EVIDENCE ห้ามใช้ความรู้ภายนอก\n"
        "faithfulness คือสัดส่วนข้ออ้างที่มีหลักฐานรองรับ และ answer_relevance "
        "คือความตรงประเด็นของคำตอบต่อคำถาม\n"
        "ให้คะแนน 0 ถึง 1 และตอบ JSON เท่านั้น: "
        '{"faithfulness":0.0,"answer_relevance":0.0,"unsupported_claims":[]}\n'
        "ห้ามให้คะแนนสูงเพียงเพราะภาษาไพเราะ และหากคำตอบปฏิเสธว่าไม่พบข้อมูล "
        "ให้ตรวจว่าการปฏิเสธสอดคล้องกับหลักฐานหรือไม่"
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer_body(answer)[:4000]}\n\n"
        f"EVIDENCE:\n{cited_text[:8000]}\n\n"
        "ประเมิน JSON เท่านั้น"
    )
    raw, metrics = assistant._llm.chat(  # type: ignore[attr-defined]
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
    result = {
        "faithfulness": max(0.0, min(1.0, float(parsed.get("faithfulness", 0.0)))),
        "answer_relevance": max(0.0, min(1.0, float(parsed.get("answer_relevance", 0.0)))),
        "unsupported_claims": parsed.get("unsupported_claims", []),
    }
    return result, metrics


def evaluate(
    dataset_path: Path,
    run_llm: bool = False,
    top_k: int = 5,
    bert_score: bool = False,
    llm_judge_enabled: bool = False,
) -> dict[str, Any]:
    dataset = read_dataset(dataset_path)
    assistant = SmartDocAssistant(settings)
    retriever = assistant.retriever()
    rows: list[dict[str, Any]] = []
    cited_evidence: list[str] = []

    for item in dataset:
        question = str(item["question"])
        target = str(item.get("target_section", ""))
        started = time.perf_counter()
        results = retriever.search(question, top_k=top_k, candidate_k=max(20, top_k * 4))
        hit = any(result.chunk.section_id == target for result in results) if target else None
        rank = next(
            (number for number, result in enumerate(results, start=1) if result.chunk.section_id == target),
            None,
        )
        row: dict[str, Any] = {
            "id": item.get("id"),
            "question": question,
            "target_section": target,
            "retrieval_hit": hit,
            "target_rank": rank,
            "retrieval_seconds": round(time.perf_counter() - started, 4),
        }
        if run_llm:
            result = assistant.answer(question, use_llm=True, retrieval_results=results)
            row.update(
                {
                    "answer": result.answer,
                    "citations": result.citations,
                    "grounded": result.grounded,
                    "answer_metrics": result.metrics,
                }
            )
            cited_ids = set(result.citations)
            unique_results = context_results(results)
            source_rows = citation_sources(results)
            cited_text = "\n".join(
                result_item.chunk.parent_text or result_item.chunk.text
                for source, result_item in zip(source_rows, unique_results)
                if source["id"] in cited_ids
            )
            cited_evidence.append(cited_text)
            row["faithfulness_heuristic"] = faithfulness_heuristic(answer_body(result.answer), cited_text)
            if llm_judge_enabled:
                try:
                    judge, judge_metrics = llm_judge(assistant, question, result.answer, cited_text)
                    row["llm_judge_faithfulness"] = judge["faithfulness"]
                    row["llm_judge_answer_relevance"] = judge["answer_relevance"]
                    row["llm_judge_unsupported_claims"] = judge["unsupported_claims"]
                    row["llm_judge_metrics"] = judge_metrics
                except Exception as exc:
                    row["llm_judge_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    semantic: dict[str, Any] = {}
    if run_llm:
        encoder = retriever._load_encoder()
        refs = [str(item.get("reference_answer", "")) for item in dataset]
        answers = [answer_body(str(row.get("answer", ""))) for row in rows]
        questions = [row["question"] for row in rows]
        ref_vectors = encoder.encode(refs, normalize_embeddings=True, convert_to_numpy=True)
        answer_vectors = encoder.encode(answers, normalize_embeddings=True, convert_to_numpy=True)
        question_vectors = encoder.encode(questions, normalize_embeddings=True, convert_to_numpy=True)
        for row, ref_vector, answer_vector, question_vector, evidence in zip(
            rows, ref_vectors, answer_vectors, question_vectors, cited_evidence
        ):
            row["sbert_cosine_reference_answer"] = round(cosine(ref_vector, answer_vector), 6)
            row["answer_relevance_cosine"] = round(cosine(question_vector, answer_vector), 6)
            row["faithfulness_semantic"] = round(
                semantic_faithfulness(str(row.get("answer", "")), evidence, encoder), 6
            )
        semantic = {
            "sbert_cosine_reference_answer_mean": mean_or_none(
                [row["sbert_cosine_reference_answer"] for row in rows]
            ),
            "answer_relevance_cosine_mean": mean_or_none(
                [row["answer_relevance_cosine"] for row in rows]
            ),
            "faithfulness_heuristic_mean": mean_or_none(
                [row["faithfulness_heuristic"] for row in rows]
            ),
            "faithfulness_semantic_mean": mean_or_none(
                [row["faithfulness_semantic"] for row in rows]
            ),
            "metric_note": "Semantic faithfulness uses the strongest BGE-M3 similarity per answer proposition; it is an evidence-support proxy, not proof of legal correctness.",
        }
        judge_faithfulness = [row["llm_judge_faithfulness"] for row in rows if "llm_judge_faithfulness" in row]
        judge_relevance = [row["llm_judge_answer_relevance"] for row in rows if "llm_judge_answer_relevance" in row]
        if judge_faithfulness:
            semantic["llm_judge_faithfulness_mean"] = mean_or_none(judge_faithfulness)
            semantic["llm_judge_answer_relevance_mean"] = mean_or_none(judge_relevance)
        elif llm_judge_enabled:
            semantic["llm_judge"] = "failed for all rows; inspect row errors"
        if bert_score:
            try:
                from bert_score import score as bert_score_fn  # type: ignore

                precision, recall, f1 = bert_score_fn(
                    answers, refs, lang="th", model_type="bert-base-multilingual-cased", verbose=False
                )
                semantic.update(
                    {
                        "bertscore_precision_mean": round(float(precision.mean()), 6),
                        "bertscore_recall_mean": round(float(recall.mean()), 6),
                        "bertscore_f1_mean": round(float(f1.mean()), 6),
                    }
                )
            except Exception as exc:
                semantic["bertscore_error"] = f"{type(exc).__name__}: {exc}"
        else:
            semantic["bertscore"] = "not requested; install bert-score and pass --bert-score"

    ranks = [row["target_rank"] for row in rows if row["target_rank"] is not None]
    answer_models = sorted(
        {
            str(row.get("answer_metrics", {}).get("model"))
            for row in rows
            if row.get("answer_metrics", {}).get("model")
        }
    )
    summary: dict[str, Any] = {
        "evaluation_version": 2,
        "dataset": str(dataset_path.resolve()),
        "queries": len(rows),
        "top_k": top_k,
        "llm_generation": run_llm,
        "llm_judge": llm_judge_enabled,
        "answer_models": answer_models,
        "retrieval": {
            "hit_at_k": round(sum(bool(row["retrieval_hit"]) for row in rows) / len(rows), 6),
            "mrr": round(sum((1.0 / rank) for rank in ranks) / len(rows), 6),
            "mean_target_rank": round(sum(ranks) / len(ranks), 6) if ranks else None,
        },
        "answer_evaluation": semantic if run_llm else "not requested; pass --run-llm",
        "rows": rows,
    }
    return summary


def comparison_table(summary: dict[str, Any]) -> list[dict[str, str]]:
    """Create a compact rubric-ready metric comparison table."""

    rows = [
        {
            "metric": "Retrieval Hit@k",
            "value": f"{summary['retrieval']['hit_at_k']:.3f}",
            "target": "1.000",
            "meaning": "target section appears in retrieved context",
        },
        {
            "metric": "MRR",
            "value": f"{summary['retrieval']['mrr']:.3f}",
            "target": "1.000",
            "meaning": "target section ranking quality",
        },
    ]
    answer = summary.get("answer_evaluation")
    if isinstance(answer, dict):
        metric_specs = [
            ("SBERT cosine / reference", "sbert_cosine_reference_answer_mean"),
            ("Answer relevance / SBERT", "answer_relevance_cosine_mean"),
            ("Faithfulness / semantic", "faithfulness_semantic_mean"),
            ("Faithfulness / lexical proxy", "faithfulness_heuristic_mean"),
            ("BERTScore Precision", "bertscore_precision_mean"),
            ("BERTScore Recall", "bertscore_recall_mean"),
            ("BERTScore F1", "bertscore_f1_mean"),
            ("Faithfulness / LLM judge", "llm_judge_faithfulness_mean"),
            ("Answer relevance / LLM judge", "llm_judge_answer_relevance_mean"),
        ]
        for label, key in metric_specs:
            if key in answer and isinstance(answer[key], (int, float)):
                rows.append(
                    {
                        "metric": label,
                        "value": f"{answer[key]:.3f}",
                        "target": "≥ 0.700",
                        "meaning": "higher is better; inspect examples with the aggregate",
                    }
                )
    return rows


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| Metric | Value | Target | Interpretation |",
        "|---|---:|---:|---|",
    ]
    lines.extend(
        f"| {row['metric']} | {row['value']} | {row['target']} | {row['meaning']} |"
        for row in rows
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SmartDoc retrieval and answers")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "eval" / "questions.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--run-llm", action="store_true")
    parser.add_argument("--bert-score", action="store_true")
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="ask Ollama to judge faithfulness/relevance; slower than embedding metrics",
    )
    parser.add_argument("--out", type=Path, default=settings.artifact_dir / "evaluation.json")
    args = parser.parse_args()
    if args.llm_judge and not args.run_llm:
        parser.error("--llm-judge requires --run-llm")
    result = evaluate(
        args.dataset,
        run_llm=args.run_llm,
        top_k=args.top_k,
        bert_score=args.bert_score,
        llm_judge_enabled=args.llm_judge,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.out.with_suffix(".csv")
    rows = result["rows"]
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row.get(key) for key in fields} for row in rows)
    table = comparison_table(result)
    report_path = args.out.with_suffix(".md")
    report_path.write_text(
        "# SmartDoc Evaluation\n\n"
        f"Dataset: `{args.dataset}`\n\n"
        + markdown_table(table)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))
    print("\nEvaluation comparison table:\n")
    print(markdown_table(table))


if __name__ == "__main__":
    main()
