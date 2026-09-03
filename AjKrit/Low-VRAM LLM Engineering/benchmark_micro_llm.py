#!/usr/bin/env python3
"""Activity 1: benchmark the three Micro-LLMs with one identical prompt."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from lab_utils import generate_with_metrics, stop_model


PROMPT = "Explain the difference between HTTP/2 and HTTP/3."
MODELS = [
    ("Qwen2.5", "qwen2.5:0.5b", "0.5B"),
    ("Llama 3.2", "llama3.2:1b", "1B"),
    ("SmolLM2", "smollm2:1.7b", "1.7B"),
]
BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-ctx", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "micro_llm_results.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    print(f"Prompt: {PROMPT}\n")

    for label, model, parameters in MODELS:
        print(f"Running {label} ({model})...", flush=True)
        stop_model(model)  # cold-load each model so wall time is comparable
        metrics = generate_with_metrics(
            model,
            PROMPT,
            num_ctx=args.num_ctx,
            keep_alive="0s",
            sample_vram=True,
        )
        row = {"label": label, "parameters": parameters, "num_ctx": args.num_ctx, **metrics}
        rows.append(row)
        print(
            f"  {metrics['wall_seconds']:.2f}s wall | "
            f"{metrics['tokens_per_second']:.1f} tok/s | "
            f"peak {metrics['peak_vram_mib']:.0f} MiB"
        )
        print(f"  Answer: {metrics['answer']}\n")

    args.output.write_text(
        json.dumps({"prompt": PROMPT, "results": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0] if key != "answer"])
        writer.writeheader()
        writer.writerows([{key: value for key, value in row.items() if key != "answer"} for row in rows])
    print(f"Saved: {args.output}\nSaved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
