# Design: FAISS and ChromaDB comparison for Thai Hybrid Retrieval

## Purpose

Create a reproducible, retrieval-only experiment that completes the
worksheet **"เปรียบเทียบ FAISS และ ChromaDB สำหรับ Hybrid RAG"**.  The
experiment compares the same Thai documents and the same normalized
`BAAI/bge-m3` embeddings in FAISS and ChromaDB.  It deliberately separates
semantic retrieval from exact metadata lookup so the strengths and limits of
each tool are observable.

This scope does not include an LLM answer generator, a chat UI, BM25, or a
production deployment.  In this worksheet, "Hybrid" means dense semantic
retrieval combined with structured metadata filtering.

## Deliverables

The implementation will add the following files under this folder:

- `hybrid_rag_comparison.py` — one command-line program that creates the
  corpus, embeddings, FAISS index, persistent Chroma collection, exercises,
  metrics, and human-readable console output.
- `requirements.txt` — runtime and test dependencies pinned only where an API
  compatibility constraint requires it.
- `test_hybrid_rag_comparison.py` — focused automated checks for dataset
  integrity, FAISS metadata filtering, and evaluation metrics.
- `README.md` — install, run, expected-output, and interpretation guidance.

At runtime the program will create only local generated artefacts, which will
be ignored by Git:

- `chroma_db_comparison/` — persistent ChromaDB storage.
- `results/` — CSV/JSON reports for semantic results, exact-code results,
  recall, and latency.

## Dataset and embedding invariant

The base corpus is the eight university-service documents and metadata defined
in the worksheet.  It has `id`, `doc_id`, `category`, and `doc_code` for each
document.  The program also includes the five extension documents requested
by Exercise 3, controlled by a command-line option so the base and expanded
experiments can be run independently.

The program loads `BAAI/bge-m3` once using `SentenceTransformer`, calls
`encode(..., normalize_embeddings=True)`, and supplies those exact vectors to
both systems.  Queries use the same model and normalization.  ChromaDB is
queried using `query_embeddings`, rather than a second internal embedding
function, so a result difference cannot be attributed to different encoding
settings.

With normalized vectors, FAISS uses `IndexFlatIP`; its inner product is cosine
similarity.  ChromaDB uses cosine HNSW distance.  The report labels FAISS
values as similarity and ChromaDB values as distance; their numeric values are
not compared as if they were the same score scale.

## Retrieval flows

### FAISS

1. Build a `faiss.IndexFlatIP` from the normalized document vectors.
2. Implement `search_faiss(query, k)` for pure semantic Top-K retrieval.
3. Keep the metadata in Python alongside vector positions.
4. Implement `search_faiss_with_code(query, doc_code, k)` as explicit
   application-side filtering.  This demonstrates that FAISS itself does not
   own or query metadata; the program must add this logic.

### ChromaDB

1. Open a `chromadb.PersistentClient` in `chroma_db_comparison/`.
2. Re-create the named collection at the beginning of a run to prevent stale
   vectors or duplicate IDs from affecting a repeatable classroom result.
3. Insert the same IDs, documents, metadata, and precomputed embeddings used
   by FAISS.
4. Implement semantic search using `query_embeddings`.
5. Implement exact-code and category-constrained semantic search using Chroma
   `where` metadata filters.

## Exercises and reports

The program runs all worksheet exercises by default and writes structured
reports as well as readable console tables.

| Worksheet work | Program output |
| --- | --- |
| Five semantic questions | FAISS and ChromaDB Top-3 document IDs/codes for each question |
| Four exact document codes | Pure FAISS, FAISS with application filter, and ChromaDB `where` results |
| Extension corpus | Corpus count and the same checks with five added documents |
| Retrieval accuracy | Recall@1, Recall@3, and Recall@5 against explicit expected document codes |
| Search latency | Mean and median vector-store retrieval time after query vectors are prepared |

Latency is reported as retrieval-only because embedding a Thai sentence can
dominate a very small index and conceal the difference between stores.  The
README will make this choice explicit and show how to measure end-to-end time
if required by the instructor.

## Error handling and repeatability

- Validate that document, metadata, embedding, and ID counts agree before an
  index or collection is built.
- Reject an invalid `k`, unknown code, or malformed metadata with clear error
  messages.
- Use deterministic IDs and a clean named Chroma collection on every run.
- Make the model-download failure actionable: the first run requires network
  access or a previously cached Hugging Face model.
- Never delete files outside the generated `chroma_db_comparison/` directory
  or the generated `results/` directory.

## Tests

Automated tests will avoid downloading the embedding model by using small,
deterministic vectors.  They will verify:

1. Base and expanded corpus records have unique IDs and required metadata.
2. FAISS pure search returns a ranked result with aligned metadata.
3. Application-side FAISS filtering returns only the requested `doc_code`.
4. Recall@K treats a correct code in the first K results as a hit.
5. A missing exact code produces an explicit empty result rather than an
   unrelated semantic match.

The full command-line experiment is additionally smoke-tested when the local
environment has the required packages and model available.

## Success criteria

The work is complete when one command creates both stores from the same
vectors, completes all three exercises, produces the comparison reports, and
passes its automated tests.  The README must make the conclusion demonstrable:
semantic results should be broadly comparable with the same embeddings, while
ChromaDB can apply an exact metadata constraint natively and FAISS requires
explicit application code.
