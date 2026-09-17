# SmartDoc Assistant — Thai Police Citizen Manual

ระบบถาม–ตอบจาก PDF ภาษาไทยแบบ evidence-first สำหรับเอกสาร `คู่มือประชาชน สำนักงานตำรวจแห่งชาติ` โดยออกแบบให้ตอบพร้อมหัวข้อและเลขหน้า และปฏิเสธคำตอบเมื่อค้นหลักฐานไม่พอ

## LINE UX and conversation history

The six-button rich menu uses `assets/richmenu_thaipol.png`, with touch areas aligned
to the artwork below its decorative header. The installer re-encodes oversized PNG
uploads to JPEG in memory; the original asset is unchanged.

```bash
.venv/bin/python setup_richmenu.py --validate-only
.venv/bin/python setup_richmenu.py
```

Installation sets the account's default rich menu, preserves existing menus, and
reuses matching menus on subsequent runs. Per-user menus can override this default.
Restart Uvicorn to enable the new webhook routing; no index rebuild is needed.
Categories show Flex carousels generated from indexed PDF headings. Subtopic
buttons and contextual quick replies use message actions and enter normal vector
RAG retrieval. Greetings/navigation skip inference and its acknowledgement.

SQLite history lives in `artifacts/chat_history.sqlite3` (private chat content;
excluded from Git, owner-only permissions). Only successfully delivered grounded
answers are stored. Defaults: four turns and 24-hour expiry, configurable through
`HISTORY_MAX_TURNS` and `HISTORY_TTL_SECONDS`. Expired records are physically removed
when history is accessed, not by a background timer. `ล้างประวัติ` / `เริ่มใหม่`
clears the current user's history. Group/room memory is isolated per member and
chat, though messages in groups remain visible to their members.

Short follow-ups such as `ต้องเตรียมอะไร` are expanded using the last source topic;
explicit new topics remain independent. History provides conversational context,
not factual evidence or reusable citations: all answers still require current PDF
retrieval. Inference is serialized for the shared local model. Navigation does not
change the stored topic; choose a subtopic or ask a new question to change it.

## Model comparison benchmark

```bash
ollama pull ministral-3:8b
.venv/bin/python benchmark_models.py --stage embeddings
.venv/bin/python benchmark_models.py --stage llms
```

Outputs are isolated in `artifacts/model_benchmark/`: JSON/CSV/Markdown summaries,
per-query retrieval rows, answer checkpoints, and frozen evidence. Production
`.env`, chunks, and FAISS index are unchanged. Run the same command to resume;
use a fresh `--out` when input hashes or settings change. The LLM stage needs the
three local Ollama models; it never sends evidence to cloud APIs.

Embedding candidates: BGE-M3, multilingual-e5-large, Qwen3-Embedding-0.6B.
E5 uses `query:` / `passage:` prefixes; Qwen uses an English retrieval instruction
for queries only. All are CPU FP32 with 512-token limits. Dense and hybrid RRF
are reported separately without reranking, to isolate embedding effects. Query
latency is measured individually, three repetitions per question, excluding
loading/index construction. Section-level retrieval scores use the existing weak
labels, not manually adjudicated chunk relevance: one target per query means
Recall@k = Hit@k; F1@5 at perfect recall is typically 0.333, not 1.0. Hit@1 and MRR
are more discriminating when Recall@5 saturates. Absolute cosine scales differ
across models and must not determine the winner.

LLMs: local Qwen 27B, Qwen 9B, and Mistral Ministral 3 8B. All get identical frozen
production BGE-M3 contexts, prompt, temperature, context window, and output token
budget. Scores include common BGE-M3 reference/relevance cosine, Thai token F1,
BERTScore P/R/F1, semantic evidence-support proxy, citation-contract rate, errors,
generation throughput and p95 latency. Citation validity is not proof of factual
support. No per-model self-judge is used: those scores would not be comparable.
Latency includes prompt+generation but excludes retrieval, warmup and LINE
delivery. With only 18 queries and one generation pass, LLM p95 is exploratory.
Avoid concurrent LINE requests while benchmarking the shared Ollama service.

Model references: [E5](https://huggingface.co/intfloat/multilingual-e5-large),
[Qwen Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B),
[Ministral](https://ollama.com/library/ministral-3:8b).
Gemma's historical scores/answers are preserved but removed from the main
comparison with `--llm-models ministral-3:8b --archive-llm gemma3:4b`.

## Source document

| Field | Value |
|---|---|
| Title | คู่มือประชาชน สำนักงานตำรวจแห่งชาติ |
| Topic | การติดต่อสถานีตำรวจ, การขออนุญาต, การป้องกันอาชญากรรม, ภัยพิบัติ, การใช้รถใช้ถนน และกฎหมายทั่วไป |
| Local file | `คู่มือประชาชน สำนักงานตำรวจแห่งชาติ.pdf` |
| Edition in document | กรกฎาคม 2558 |
| Link printed in source PDF | https://www.facebook.com/การมีส่วนร่วมของประชาชน |
| Important limitation | เป็นเอกสารประวัติศาสตร์ ไม่ใช่การยืนยันกฎหมาย/ขั้นตอนปัจจุบัน |

## Architecture

```text
PDF
 └─ layout-aware extraction → Unicode cleanup → repeated header/footer removal
    └─ section-aware parent passages (2,400 chars)
       └─ Thai recursive child chunks (900 chars, 120 overlap)
       ├─ BGE-M3 normalized embeddings → FAISS IndexFlatIP
       └─ Thai newmm tokens → BM25
             └─ weighted RRF → optional CrossEncoder rerank
                 └─ dynamic top-k + context budget
                     └─ Ollama JSON answer → citation validation → LINE push
```

The index stores FAISS natively and stores child plus parent metadata as JSON.
Each retrieval child keeps its larger parent passage, parent page range and
section identity.  The answer layer deduplicates overlapping child hits and
feeds one coherent parent passage per citation to the LLM.  It does not load
an unsafe LangChain pickle file.

PyMuPDF is the primary parser.  The pinned version uses layout-aware text
blocks and `find_tables()` when tables are present, while `pdftotext -layout`
remains a deterministic fallback.  Invisible PDF characters including zero
width spaces/joiners, BOMs and soft hyphens are removed before section
detection, BM25 tokenisation and chunking.

## Install and build

```bash
cd Project2_SmartDoc_Assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 build_index.py
```

The first build downloads the embedding model and can take several minutes.
Generated files are placed under `artifacts/index/` and include `index.faiss`,
`chunks.json`, `manifest.json`, `build_metrics.json`, and a page inventory.

## Run a query

```bash
python3 cli.py "คนหายต้องรอครบ 24 ชั่วโมงก่อนแจ้งความหรือไม่"
python3 cli.py "รถหรือเรือหายควรโทรหมายเลขใด" --retrieve-only --json
```

Start Ollama separately before a full answer:

```bash
ollama serve
ollama run hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

Production now uses `OLLAMA_MODEL=qwen3.5:9b` in `.env`; 27B is an offline comparator.

## Why FAISS + BM25 instead of only one vector store?

The retriever intentionally uses both channels:

- Dense search catches paraphrases such as “ญาติหายตัวไป” → `1.4 แจ้งความคนหาย`.
- BM25 catches exact terms, phone numbers, section numbers, document names, and legal phrases.
- RRF combines ranks, so incompatible raw score scales are not compared directly.
- The weighted dense/BM25 score is retained for audit and tuning.
- `RERANK_MODEL=BAAI/bge-reranker-v2-m3` enables the optional Cross-Encoder stage.

The existing 10,000-row local benchmark in this workspace measured FAISS
`IndexFlatIP` at about 1.56 ms mean retrieval latency and ChromaDB HNSW at about
1.28 ms, excluding query embedding. ChromaDB used about 318 MB persisted
storage and 25.3 s to build, while the exact FAISS build took about 0.02 s.
For this static 130-page document, FAISS is simpler and reproducible; ChromaDB
would be a reasonable replacement when collection-level metadata filtering and
frequent updates matter more than minimal local artifacts.

## Qwen3.8-27B latency decision

The workstation probe already recorded the following for
`hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M`:

| Model | Hardware | Output speed | Wall time in probe |
|---|---|---:|---:|
| Qwen3.8-27B Q4 | Ryzen 7 7735HS, 30 GiB RAM, RX 7600S 8 GiB, GPU/RAM split | 3.18–3.39 tok/s | 44–69 s |
| Qwen3.5:9b | Same machine | 34.67–35.05 tok/s | 3–6 s |

The 27B model is usable for quality-focused offline answering, but it is too
slow to hold LINE's webhook request open. The webhook therefore replies with a
short acknowledgement immediately and sends the final grounded answer through
LINE Push Message after inference completes. `keep_alive=15m`, JSON output,
temperature 0, short context, and a 420-token answer limit reduce latency.

## Best features of the chat

1. **Evidence-first answer** — the model receives only retrieved passages; every answer must contain `[S1]`, `[S2]`, etc.
2. **Source display** — answer footer includes document title, topic, PDF page, and printed page when available.
3. **Strict fallback** — weak retrieval or invalid/missing citations returns `ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร`.
4. **Dynamic retrieval** — short questions retrieve fewer chunks; long/multi-part questions retrieve more, within a fixed context budget.
5. **Hybrid audit trail** — dense score, BM25 score, dense/sparse rank, RRF score, and optional reranker score are available in JSON.
6. **Thai handling** — Unicode cleanup, Thai `newmm` BM25 tokenisation, section-aware chunks, and page provenance.
7. **LINE reliability** — signature verification, deduplication, immediate acknowledgement, background inference, push response, and bounded message splitting.
8. **Offline privacy** — embeddings and LLM run locally through Ollama; LINE only receives the final response when the webhook is enabled.

## LINE webhook

Put rotated values in `.env` only:

```dotenv
CHANNEL_SECRET=...
CHANNEL_ACCESS_TOKEN=...
```

Run:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Expose `POST /callback` through an HTTPS tunnel and set that URL in LINE
Developers Console. `GET /health` reports only non-secret readiness fields.

The credentials pasted in the conversation are considered exposed. Rotate the
Channel secret and access token before using this deployment; this repository
does not contain them.

## Evaluation

The test set is `eval/questions.jsonl` with 18 Thai Q&A pairs. Retrieval-only:

```bash
python3 evaluate.py
```

Full evaluation, including real Ollama answer generation, SBERT cosine
similarity, semantic faithfulness, answer relevance and a generated comparison
table:

```bash
OLLAMA_MODEL=qwen3.5:9b OLLAMA_TIMEOUT_SECONDS=180 MAX_ANSWER_TOKENS=240 \
  python3 evaluate.py --run-llm --bert-score
```

For an additional local LLM-as-judge score (slower because it makes one extra
Ollama call per test question):

```bash
python3 evaluate.py --run-llm --bert-score --llm-judge
```

`evaluate.py` writes JSON, CSV and a Markdown metric table under `artifacts/`.
The lexical faithfulness score is explicitly labelled a heuristic.  The
semantic and LLM-judge scores are evidence-support metrics, not proof of
current legal correctness; human review is still required.

Latest 18-question full run: Hit@5 `1.000`, MRR `1.000`, SBERT/reference
`0.833`, answer relevance `0.756`, semantic faithfulness `1.000`, and
BERTScore P/R/F1 `0.766/0.850/0.805`. The additional local LLM judge scored
faithfulness/relevance `0.981/1.000`. All 18 answers were grounded and had a
valid citation. The full run used local `qwen3.5:9b` for practical runtime;
the production `.env` now uses Qwen 9B, selected for local LINE latency.

## Rubric coverage

| Rubric | Implementation |
|---|---|
| PDF ingestion/chunking | `ingest.py`: PyMuPDF layout/table extraction, zero-width cleanup, repeated furniture removal, section-aware parent documents, recursive child splitting, overlap and page metadata |
| Hybrid retrieval | `retriever.py`: FAISS + Thai BM25 + weighted RRF + optional CrossEncoder |
| Dynamic top-k/embedding | `dynamic_top_k`, normalized `BAAI/bge-m3`, candidate over-fetch, context budget |
| Prompt/guardrail | `llm.py`: context-only system prompt, JSON schema, citation allow-list, strict fallback |
| Quantitative evaluation | `evaluate.py` + `eval/questions.jsonl`: Hit@k, MRR, SBERT cosine, answer relevance, semantic faithfulness, optional BERTScore P/R/F1, optional LLM judge, Markdown comparison table |
| Reproducibility | exact top-level dependency pins, `.env.example`, manifest version, SHA-256 source check, unit tests |

## Tests

```bash
python3 -m unittest discover -s tests -p 'test*.py' -v
```
