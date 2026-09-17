"""Command-line interface for retrieval-only and full SmartDoc answers."""

from __future__ import annotations

import argparse
import json

from assistant import SmartDocAssistant
from config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="ถามคู่มือประชาชนด้วย Hybrid RAG")
    parser.add_argument("question", nargs="?", help="คำถามภาษาไทย")
    parser.add_argument("--retrieve-only", action="store_true")
    parser.add_argument("--json", action="store_true", help="พิมพ์ผลเป็น JSON")
    args = parser.parse_args()
    question = (args.question or input("คำถาม: ")).strip()
    result = SmartDocAssistant(settings).answer(question, use_llm=not args.retrieve_only)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.answer)
        print("\n--- metrics ---")
        print(json.dumps(result.metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

