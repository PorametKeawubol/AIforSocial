"""Run the six-intent lab evaluation and print a Markdown-ready report."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from intent_classifier import (
    INTENT_LABELS,
    LAB_EDGE_CASES,
    LAB_TEST_CASES,
    IntentClassifier,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lexical",
        action="store_true",
        help="use the offline lexical fallback instead of Sentence-BERT",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override the intent acceptance threshold",
    )
    args = parser.parse_args()

    classifier = IntentClassifier(
        semantic_enabled=not args.lexical,
        threshold=args.threshold,
    )
    correct = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    print("| # | ข้อความ | Intent จริง | Intent ที่ทำนาย | Similarity สูงสุด | ผล | คะแนนราย Intent |")
    print("|---:|---|---|---|---:|:---:|---|")
    for index, (text, expected) in enumerate(LAB_TEST_CASES, start=1):
        prediction = classifier.detect(text)
        is_correct = prediction.intent == expected
        correct += is_correct
        confusion[expected][prediction.intent] += 1
        score_text = ", ".join(
            f"{intent}={score:.3f}"
            for intent, score in prediction.scores.items()
        )
        mark = "✓" if is_correct else "✗"
        print(
            f"| {index} | {text} | {expected} | {prediction.intent} | "
            f"{prediction.score:.3f} | {mark} | {score_text} |"
        )

    total = len(LAB_TEST_CASES)
    print(f"\nBackend: `{classifier.detect(LAB_TEST_CASES[0][0]).backend}`")
    print(f"Threshold: `{classifier.threshold:.2f}`")
    print(f"Accuracy: `{correct}/{total} = {correct / total:.2%}`")
    print("\nConfusion summary:")
    for expected in INTENT_LABELS:
        print(f"- `{expected}`: {dict(confusion[expected])}")

    edge_correct = 0
    print("\nBoundary cases (not included in the six-intent accuracy):")
    print("| ข้อความ | Intent จริง | Intent ที่ทำนาย | Similarity | ผล |")
    print("|---|---|---|---:|:---:|")
    for text, expected in LAB_EDGE_CASES:
        prediction = classifier.detect(text)
        is_correct = prediction.intent == expected
        edge_correct += is_correct
        mark = "✓" if is_correct else "✗"
        print(
            f"| {text} | {expected} | {prediction.intent} | "
            f"{prediction.score:.3f} | {mark} |"
        )
    print(f"Boundary accuracy: `{edge_correct}/{len(LAB_EDGE_CASES)}`")


if __name__ == "__main__":
    main()
