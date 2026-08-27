# Performance report

Measured on 2026-08-28 with Python 3.12.3 on an AMD Ryzen 7 7735HS
(8 cores / 16 threads), using the checked-in 1,715-product Mercular snapshot:

```bash
python scripts/benchmark.py
```

| Metric | Result |
|---|---:|
| Iterations | 100 |
| Mean | 122.402 ms |
| p50 | 108.024 ms |
| p95 | 171.022 ms |
| Max | 181.217 ms |
| Rubric target | <=1,500 ms |

The benchmark warms imports/caches, then measures the local user-command path:
NLP parsing, hard-filtered retrieval, randomized Top 5 selection, and Flex
payload rendering. It reads the same local snapshot as the webhook; no scrape or
source-site request is performed while a user waits.

External LINE network latency is intentionally excluded because it depends on
the deployment network and credentials. Production reuses the LINE API client's
connection pool, has explicit 2-second connect and 5-second read timeouts, and
returns non-2xx on retryable delivery failures so LINE can redeliver.
