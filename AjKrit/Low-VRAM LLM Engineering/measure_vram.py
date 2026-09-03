#!/usr/bin/env python3
"""Activity 2: collect AMD VRAM readings before, during, and after one LLM request."""

from __future__ import annotations

import json
from pathlib import Path

from lab_utils import bytes_to_mib, generate_with_metrics, stop_model, vram_used_bytes, wait_for_vram_settle


BASE_DIR = Path(__file__).resolve().parent
MODEL = "llama3.2:1b"
PROMPT = "Explain Retrieval-Augmented Generation in two short sentences."


def main() -> int:
    stop_model(MODEL)
    before = vram_used_bytes()
    metrics = generate_with_metrics(
        MODEL,
        PROMPT,
        num_ctx=1024,
        keep_alive="10m",
        sample_vram=True,
    )
    loaded = vram_used_bytes()
    stop_model(MODEL)
    after_stop = wait_for_vram_settle()

    result = {
        "model": MODEL,
        "prompt": PROMPT,
        "before_mib": bytes_to_mib(before),
        "loaded_after_generate_mib": bytes_to_mib(loaded),
        "peak_during_generation_mib": metrics["peak_vram_mib"],
        "after_stop_mib": bytes_to_mib(after_stop),
        "generation": metrics,
    }
    output = BASE_DIR / "vram_measurement.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== VRAM measurement (MiB) ===")
    print(f"Before run          : {result['before_mib']:.1f}")
    print(f"Model loaded        : {result['loaded_after_generate_mib']:.1f}")
    print(f"Peak while generating: {result['peak_during_generation_mib']:.1f}")
    print(f"After ollama stop   : {result['after_stop_mib']:.1f}")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
