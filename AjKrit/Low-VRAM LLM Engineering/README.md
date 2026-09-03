# Low-VRAM LLM Engineering

Implementation and measured results for the supplied Low-VRAM LLM Engineering
lab.  It uses Ollama locally, `ChatOllama` for the LangChain task, ChromaDB for
local retrieval, and `rocm-smi` (when available) for AMD VRAM telemetry.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
ollama serve                           # use another terminal after it is ready
ollama pull qwen2.5:0.5b
ollama pull llama3.2:1b
ollama pull smollm2:1.7b
```

## Run the lab

```bash
python benchmark_micro_llm.py          # Activity 1
python measure_vram.py                 # Activity 2
python app.py --num-ctx 1024           # Part 3 / LangChain + Ollama
python context_benchmark.py            # Activity 4
python rag_pipeline.py                 # Activity 5
python compare_llm_rag.py              # Activity 6
python routed_rag.py                   # Part 7 challenge
```

## Strict worksheet run

For a run that follows the worksheet's specified models, prompts, context
settings, retrieval setting, and plain Qwen router without extra guardrails:

```bash
python app_lab.py
python strict_lab_runner.py
```

This writes `strict_lab_results.json`; the matching tables and analysis are in
`LAB_REPORT.md`.

`routed_rag.py` first asks Qwen 0.5B to route the query.  It also has a narrow
networking-domain guardrail: a false-positive LLM routing decision cannot send
a query with no HTTP/2, HTTP/3, QUIC, TCP, UDP, or “transport protocol” term
to the knowledge branch.  This keeps a small router model from wasting RAG
resources on obvious small-talk requests.

## Files

- `benchmark_micro_llm.py` — same-prompt Micro-LLM speed/VRAM benchmark.
- `measure_vram.py` — before/load/generation/unload VRAM capture.
- `app.py` — `langchain_ollama.ChatOllama` example.
- `context_benchmark.py` — `num_ctx` 512, 1024, and 2048 comparison.
- `rag_pipeline.py` — ChromaDB retrieval + Llama 1B generation.
- `compare_llm_rag.py` — direct LLM vs source-grounded RAG test.
- `routed_rag.py` — low-VRAM query router.
- `LAB_REPORT.md` — tables, analysis answers, and important observed limits.

Generated `.json` and `.csv` files preserve the actual run outputs and timing
metadata for the report.
