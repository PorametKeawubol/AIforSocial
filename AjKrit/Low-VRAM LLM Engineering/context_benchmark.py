#!/usr/bin/env python3
"""Activity 4: measure the effect of num_ctx on a local Llama 1B request."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from lab_utils import generate_with_metrics, stop_model


BASE_DIR = Path(__file__).resolve().parent
MODEL = "llama3.2:1b"
PROMPT = "Explain HTTP/3 in simple terms, in no more than three sentences."
CONTEXT_SIZES = (512, 1024, 2048)


def main() -> int:
    rows: list[dict[str, object]] = []
    for num_ctx in CONTEXT_SIZES:
        print(f"Running num_ctx={num_ctx}...", flush=True)
        stop_model(MODEL)
        metrics = generate_with_metrics(
            MODEL, PROMPT, num_ctx=num_ctx, keep_alive="0s", sample_vram=True
        )
        row = {"num_ctx": num_ctx, **metrics}
        rows.append(row)
        print(
            f"  wall={metrics['wall_seconds']:.2f}s | "
            f"peak={metrics['peak_vram_mib']:.0f} MiB | {metrics['tokens_per_second']:.1f} tok/s"
        )

    json_path = BASE_DIR / "context_size_results.json"
    json_path.write_text(
        json.dumps({"model": MODEL, "prompt": PROMPT, "results": rows}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    csv_path = json_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0] if key != "answer"])
        writer.writeheader()
        writer.writerows([{key: value for key, value in row.items() if key != "answer"} for row in rows])
    print(f"Saved: {json_path}\nSaved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
