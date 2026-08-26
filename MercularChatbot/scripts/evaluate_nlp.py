#!/usr/bin/env python3
"""Evaluate Mercular intent/entities and enforce the rubric's strict >85% gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable, Mapping


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from nlp import CommandEntities, ThaiCommandParser, compact_text  # noqa: E402


DEFAULT_DATASET = PROJECT_DIR / "data" / "nlp_evaluation.json"
ENTITY_FIELDS = (
    "category",
    "brands",
    "min_price",
    "max_price",
    "min_price_inclusive",
    "max_price_inclusive",
    "in_stock",
    "sort",
    "query",
)
REQUIRED_KINDS = {"basic", "colloquial", "typo", "multi_condition", "ambiguous", "no_match"}
PASS_THRESHOLD = 0.85


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    total: int
    intent_correct: int
    entity_fields_correct: int
    entity_fields_total: int
    entity_true_positive: int
    entity_false_positive: int
    entity_false_negative: int
    joint_correct: int
    latency_ms: tuple[float, ...]

    @property
    def intent_accuracy(self) -> float:
        return self.intent_correct / self.total if self.total else 0.0

    @property
    def entity_field_accuracy(self) -> float:
        return self.entity_fields_correct / self.entity_fields_total if self.entity_fields_total else 0.0

    @property
    def entity_precision(self) -> float:
        denominator = self.entity_true_positive + self.entity_false_positive
        return self.entity_true_positive / denominator if denominator else 1.0

    @property
    def entity_recall(self) -> float:
        denominator = self.entity_true_positive + self.entity_false_negative
        return self.entity_true_positive / denominator if denominator else 1.0

    @property
    def entity_f1(self) -> float:
        denominator = self.entity_precision + self.entity_recall
        return 2 * self.entity_precision * self.entity_recall / denominator if denominator else 0.0

    @property
    def joint_accuracy(self) -> float:
        return self.joint_correct / self.total if self.total else 0.0

    @property
    def mean_latency_ms(self) -> float:
        return statistics.fmean(self.latency_ms) if self.latency_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latency_ms:
            return 0.0
        ordered = sorted(self.latency_ms)
        index = max(0, math_ceil(0.95 * len(ordered)) - 1)
        return ordered[index]

    @property
    def passed(self) -> bool:
        # "มากกว่า 85%" in the rubric is strict, so exactly 85% does not pass.
        return (
            self.intent_accuracy > PASS_THRESHOLD
            and self.entity_field_accuracy > PASS_THRESHOLD
            and self.entity_f1 > PASS_THRESHOLD
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "intent_accuracy": self.intent_accuracy,
            "entity_field_accuracy": self.entity_field_accuracy,
            "entity_precision": self.entity_precision,
            "entity_recall": self.entity_recall,
            "entity_f1": self.entity_f1,
            "joint_accuracy": self.joint_accuracy,
            "latency_ms": {
                "mean": self.mean_latency_ms,
                "p95": self.p95_latency_ms,
                "max": max(self.latency_ms, default=0.0),
            },
            "passed": self.passed,
        }


def math_ceil(value: float) -> int:
    integer = int(value)
    return integer if value == integer else integer + 1


def load_cases(path: Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    """Load and validate the labelled JSON evaluation corpus."""

    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list) or len(raw) < 50:
        raise ValueError("evaluation dataset must contain at least 50 cases")
    cases: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"case {index} must be an object")
        identifier = str(item.get("id", ""))
        if not identifier or identifier in identifiers:
            raise ValueError(f"case {index} has a missing/duplicate id")
        identifiers.add(identifier)
        if not str(item.get("text", "")).strip() and item.get("text") != "":
            raise ValueError(f"case {identifier} is missing text")
        if not str(item.get("intent", "")):
            raise ValueError(f"case {identifier} is missing intent")
        if not isinstance(item.get("entities", {}), dict):
            raise ValueError(f"case {identifier} entities must be an object")
        cases.append(item)
    missing_kinds = REQUIRED_KINDS - {str(case.get("kind")) for case in cases}
    if missing_kinds:
        raise ValueError(f"dataset is missing required kinds: {sorted(missing_kinds)}")
    return cases


def _expected_entities(value: Mapping[str, Any]) -> dict[str, Any]:
    defaults = CommandEntities().to_dict()
    unknown = set(value) - set(ENTITY_FIELDS)
    if unknown:
        raise ValueError(f"unknown entity fields: {sorted(unknown)}")
    defaults.update(value)
    defaults["brands"] = list(defaults.get("brands") or [])
    return defaults


def _comparable(field: str, value: Any) -> Any:
    if field == "brands":
        return tuple(sorted(compact_text(item) for item in (value or [])))
    if field in {"category", "query"}:
        return compact_text(value or "") or None
    if field in {"min_price", "max_price"}:
        return None if value is None else round(float(value), 4)
    return value


def _entity_atoms(entities: Mapping[str, Any]) -> set[tuple[str, Any]]:
    atoms: set[tuple[str, Any]] = set()
    for field in ENTITY_FIELDS:
        value = _comparable(field, entities.get(field))
        if field == "brands":
            atoms.update((field, brand) for brand in value)
        elif value not in (None, "", ()):
            atoms.add((field, value))
    return atoms


def evaluate_cases(
    cases: Iterable[Mapping[str, Any]],
    parser: ThaiCommandParser | None = None,
) -> tuple[EvaluationMetrics, list[dict[str, Any]]]:
    """Evaluate cases and return aggregate metrics plus per-case diagnostics."""

    command_parser = parser or ThaiCommandParser()
    rows: list[dict[str, Any]] = []
    intent_correct = entity_fields_correct = entity_fields_total = joint_correct = 0
    true_positive = false_positive = false_negative = 0
    latencies: list[float] = []

    for case in cases:
        started = time.perf_counter_ns()
        predicted = command_parser.parse(case.get("text"))
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        latencies.append(elapsed_ms)
        expected_entities = _expected_entities(case.get("entities", {}))
        actual_entities = predicted.entities.to_dict()
        field_results = {
            field: _comparable(field, actual_entities[field])
            == _comparable(field, expected_entities[field])
            for field in ENTITY_FIELDS
        }
        fields_correct = sum(field_results.values())
        entity_fields_correct += fields_correct
        entity_fields_total += len(ENTITY_FIELDS)
        expected_atoms = _entity_atoms(expected_entities)
        actual_atoms = _entity_atoms(actual_entities)
        true_positive += len(expected_atoms & actual_atoms)
        false_positive += len(actual_atoms - expected_atoms)
        false_negative += len(expected_atoms - actual_atoms)
        correct_intent = predicted.intent == case["intent"]
        intent_correct += correct_intent
        correct_joint = correct_intent and all(field_results.values())
        joint_correct += correct_joint
        rows.append(
            {
                "id": case["id"],
                "kind": case.get("kind", ""),
                "text": case.get("text", ""),
                "expected_intent": case["intent"],
                "predicted_intent": predicted.intent,
                "expected_entities": expected_entities,
                "predicted_entities": actual_entities,
                "intent_correct": correct_intent,
                "entity_fields_correct": fields_correct,
                "joint_correct": correct_joint,
                "latency_ms": elapsed_ms,
            }
        )

    metrics = EvaluationMetrics(
        total=len(rows),
        intent_correct=intent_correct,
        entity_fields_correct=entity_fields_correct,
        entity_fields_total=entity_fields_total,
        entity_true_positive=true_positive,
        entity_false_positive=false_positive,
        entity_false_negative=false_negative,
        joint_correct=joint_correct,
        latency_ms=tuple(latencies),
    )
    return metrics, rows


def _print_report(metrics: EvaluationMetrics, rows: list[dict[str, Any]]) -> None:
    print("Mercular Thai/English NLP evaluation")
    print(f"Cases:                  {metrics.total}")
    print(f"Intent accuracy:        {metrics.intent_accuracy:.2%}")
    print(f"Entity field accuracy:  {metrics.entity_field_accuracy:.2%}")
    print(f"Entity precision:       {metrics.entity_precision:.2%}")
    print(f"Entity recall:          {metrics.entity_recall:.2%}")
    print(f"Entity F1:              {metrics.entity_f1:.2%}")
    print(f"Joint exact accuracy:   {metrics.joint_accuracy:.2%}")
    print(f"Latency mean / p95/max: {metrics.mean_latency_ms:.3f} / {metrics.p95_latency_ms:.3f} / {max(metrics.latency_ms, default=0):.3f} ms")
    print(f"Rubric gate (>85%):     {'PASS' if metrics.passed else 'FAIL'}")

    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_kind[row["kind"]]["total"] += 1
        by_kind[row["kind"]]["intent"] += int(row["intent_correct"])
        by_kind[row["kind"]]["joint"] += int(row["joint_correct"])
    print("\nBreakdown by case kind:")
    for kind in sorted(by_kind):
        count = by_kind[kind]
        print(
            f"- {kind}: intent {count['intent']}/{count['total']}, "
            f"joint {count['joint']}/{count['total']}"
        )

    failures = [row for row in rows if not row["joint_correct"]]
    if failures:
        print("\nNon-joint cases:")
        for row in failures:
            print(
                f"- {row['id']} {row['text']!r}: "
                f"intent {row['expected_intent']} -> {row['predicted_intent']}; "
                f"entities {row['expected_entities']} -> {row['predicted_entities']}"
            )


def main(argv: list[str] | None = None) -> int:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    argument_parser.add_argument("--json", action="store_true", help="emit aggregate JSON")
    args = argument_parser.parse_args(argv)
    try:
        cases = load_cases(args.dataset)
        metrics, rows = evaluate_cases(cases)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Evaluation error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_report(metrics, rows)
    return 0 if metrics.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
