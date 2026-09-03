"""Shared utilities for the Low-VRAM LLM Engineering lab.

The lab uses the local Ollama server only.  GPU-memory collection is optional:
on AMD/ROCm systems it samples ``rocm-smi``; on other systems it returns None.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any

from ollama import Client


OLLAMA_HOST = "http://127.0.0.1:11434"


def ollama_client() -> Client:
    return Client(host=OLLAMA_HOST)


def vram_used_bytes(gpu_index: int = 0) -> int | None:
    """Read used VRAM for one AMD GPU, or None when ROCm tools are unavailable."""
    try:
        output = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            check=False,
            text=True,
            capture_output=True,
            timeout=5,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    match = re.search(
        rf"GPU\[{gpu_index}\].*?VRAM Total Used Memory \(B\):\s*(\d+)",
        output,
        flags=re.DOTALL,
    )
    return int(match.group(1)) if match else None


def bytes_to_mib(value: int | None) -> float | None:
    return None if value is None else value / (1024 * 1024)


class VRAMSampler:
    """Sample VRAM in a lightweight background thread while a request runs."""

    def __init__(self, interval_seconds: float = 0.25, gpu_index: int = 0) -> None:
        self.interval_seconds = interval_seconds
        self.gpu_index = gpu_index
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        value = vram_used_bytes(self.gpu_index)
        if value is not None:
            self.samples.append((time.perf_counter(), value))

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 2)
        self._sample_once()

    @property
    def peak_bytes(self) -> int | None:
        return max((value for _, value in self.samples), default=None)


def response_to_dict(response: Any) -> dict[str, Any]:
    """Support both dict and Pydantic responses from different ollama versions."""
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return dict(response)


def generate_with_metrics(
    model: str,
    prompt: str,
    *,
    num_ctx: int = 1024,
    num_gpu: int = 99,
    num_thread: int = 4,
    keep_alive: str = "0s",
    sample_vram: bool = False,
) -> dict[str, Any]:
    """Generate once and return its answer, Ollama timings, and optional VRAM peak."""
    sampler = VRAMSampler() if sample_vram else None
    if sampler:
        sampler.start()
    started = time.perf_counter()
    try:
        raw_response = ollama_client().generate(
            model=model,
            prompt=prompt,
            options={
                "num_ctx": num_ctx,
                "num_gpu": num_gpu,
                "num_thread": num_thread,
                "temperature": 0,
            },
            keep_alive=keep_alive,
        )
    finally:
        if sampler:
            sampler.stop()
    wall_seconds = time.perf_counter() - started
    response = response_to_dict(raw_response)

    eval_count = int(response.get("eval_count") or 0)
    eval_duration_ns = int(response.get("eval_duration") or 0)
    eval_seconds = eval_duration_ns / 1_000_000_000
    return {
        "model": model,
        "answer": str(response.get("response", "")).strip(),
        "wall_seconds": wall_seconds,
        "total_seconds": int(response.get("total_duration") or 0) / 1_000_000_000,
        "load_seconds": int(response.get("load_duration") or 0) / 1_000_000_000,
        "prompt_tokens": int(response.get("prompt_eval_count") or 0),
        "generated_tokens": eval_count,
        "generation_seconds": eval_seconds,
        "tokens_per_second": (eval_count / eval_seconds) if eval_seconds else None,
        "peak_vram_mib": bytes_to_mib(sampler.peak_bytes) if sampler else None,
    }


def stop_model(model: str) -> None:
    """Unload a model without deleting it, equivalent to leaving the Ollama chat."""
    subprocess.run(["ollama", "stop", model], check=False, capture_output=True, text=True)


def wait_for_vram_settle(timeout_seconds: float = 15.0) -> int | None:
    """Return a stable post-unload reading, if ROCm memory telemetry is available."""
    last = vram_used_bytes()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        time.sleep(0.5)
        current = vram_used_bytes()
        if current is not None and last is not None and abs(current - last) < 4 * 1024 * 1024:
            return current
        last = current
    return last


def run_with_sampler(action: Callable[[], dict[str, Any]]) -> tuple[dict[str, Any], int | None]:
    """Execute an action while sampling VRAM; useful for non-Ollama calls too."""
    sampler = VRAMSampler()
    sampler.start()
    try:
        result = action()
    finally:
        sampler.stop()
    return result, sampler.peak_bytes
