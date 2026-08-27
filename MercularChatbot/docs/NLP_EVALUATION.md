# NLP evaluation report

Measured on 2026-08-28 with Python 3.12.3:

```bash
python scripts/evaluate_nlp.py
```

| Metric | Result |
|---|---:|
| Labeled commands | 177 |
| Intent accuracy | 100.00% |
| Entity-field accuracy | 100.00% |
| Entity precision / recall / F1 | 100.00% / 100.00% / 100.00% |
| Joint exact accuracy | 100.00% |
| Mean / p95 / max parser latency | 3.961 / 5.696 / 27.450 ms |
| Rubric gate | PASS (>85%) |

The corpus covers 50 basic, 35 colloquial, 33 typo, 39 multi-condition,
10 ambiguous, and 10 no-match cases. It includes Thai and English aliases,
Thai digits, `k` notation, strict and inclusive price bounds, stock negation,
catalog subtypes, misspellings, and commands found during adversarial live-catalog
review.

This is a checked-in, project-local regression corpus—not an independent claim
about general-language accuracy. The JSON labels and evaluator are reproducible
in `data/nlp_evaluation.json` and `scripts/evaluate_nlp.py`.
