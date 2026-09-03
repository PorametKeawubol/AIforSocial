#!/usr/bin/env python3
"""Part 3: invoke a local Ollama model through LangChain's ChatOllama."""

from __future__ import annotations

import argparse
import time

from langchain_ollama import ChatOllama


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama3.2:1b")
    parser.add_argument("--num-ctx", type=int, default=1024)
    parser.add_argument("--num-gpu", type=int, default=99, help="Layers to offload; 99 means all eligible layers.")
    parser.add_argument("--num-thread", type=int, default=4)
    parser.add_argument("--prompt", default="Explain HTTP/3 in simple terms.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    llm = ChatOllama(
        model=args.model,
        num_gpu=args.num_gpu,
        num_ctx=args.num_ctx,
        num_thread=args.num_thread,
        temperature=0,
        keep_alive="0s",
    )
    started = time.perf_counter()
    response = llm.invoke(args.prompt)
    elapsed = time.perf_counter() - started
    print(f"Model: {args.model} | num_ctx={args.num_ctx} | {elapsed:.2f}s")
    print(response.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
