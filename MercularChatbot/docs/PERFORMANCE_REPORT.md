# Performance report

Measured on 2026-08-27 with Python 3.12.3 on an AMD Ryzen 7 7735HS
(8 cores / 16 threads), using the checked-in 1,715-product Mercular snapshot:

```bash
python scripts/benchmark.py
```

| Metric | Result |
|---|---:|
| Iterations | 100 |
| Mean | 196.244 ms |
| p50 | 148.533 ms |
| p95 | 430.152 ms |
| Max | 479.049 ms |
| Rubric target | <=1,500 ms |

The benchmark warms imports/caches, then measures the local user-command path:
NLP parsing, hard-filtered retrieval, randomized Top 5 selection, and Flex
payload rendering. It reads the same local snapshot as the webhook; no scrape or
source-site request is performed while a user waits.

External LINE network latency is intentionally excluded because it depends on
the deployment network and credentials. Production sends have explicit 2-second
connect and 5-second read timeouts, while retryable delivery failures make the
webhook return non-2xx so LINE can redeliver.
