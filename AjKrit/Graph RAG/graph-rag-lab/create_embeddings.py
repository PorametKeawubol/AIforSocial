"""Lab 5, Tasks 7–11: embed chunks, connect entities, and create a vector index.

Run create_graph.py first. Re-running resumes from verified embeddings already
stored in Neo4j. The first run downloads the Sentence Transformers model.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from common import (
    DATABASE, EMBEDDING_MODEL, INDEX_NAME, ROOT, batches, get_driver,
    load_model, read_json, write_json,
)
from create_graph import dataset_hash

DIMENSIONS = 1024


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_graph(session, digest, chunks):
    record = session.run(
        "MATCH (m:Lab5Metadata {id: 'dataset'}) "
        "RETURN m.sha256 AS sha256, m.graph_complete AS graph_complete"
    ).single()
    if not record or record["sha256"] != digest or not record["graph_complete"]:
        raise ValueError("Run create_graph.py successfully with this prepared.json first.")
    entity_ids = sorted({chunk["entity_id"] for chunk in chunks})
    count = session.run(
        "MATCH (n:AttackEntity {lab: 'lab5', dataset_sha256: $digest}) "
        "WHERE n.id IN $ids RETURN count(n) AS count",
        ids=entity_ids,
        digest=digest,
    ).single(strict=True)["count"]
    if count != len(entity_ids):
        raise ValueError("Some chunks reference entities missing from this graph snapshot.")
    conflicts = session.run(
        "MATCH (c:Lab5BGEChunk) WHERE c.id IN $ids "
        "AND (coalesce(c.lab, '') <> 'lab5' "
        "OR coalesce(c.dataset_sha256, '') <> $digest) "
        "RETURN count(c) AS count",
        ids=[chunk["id"] for chunk in chunks], digest=digest,
    ).single(strict=True)["count"]
    if conflicts:
        raise ValueError("Chunk ids already belong to another dataset; no chunks were changed.")


def index_details(session):
    return session.run(
        "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, "
        "state, populationPercent, options, failureMessage "
        "WHERE name = $name RETURN *",
        name=INDEX_NAME,
    ).single()


def verify_index(record):
    if not record:
        raise RuntimeError(f"Index {INDEX_NAME} was not created; check for another index on this schema.")
    config = record["options"].get("indexConfig", {})
    if (
        record["type"] != "VECTOR"
        or record["entityType"] != "NODE"
        or record["labelsOrTypes"] != ["Lab5BGEChunk"]
        or record["properties"] != ["embedding"]
        or config.get("vector.dimensions") != DIMENSIONS
        or str(config.get("vector.similarity_function", "")).lower() != "cosine"
    ):
        raise ValueError(f"Existing {INDEX_NAME} has incompatible label, property, dimensions, or similarity.")


def store_batch(tx, rows, digest):
    return tx.run(
        "UNWIND $rows AS row "
        "MATCH (entity:AttackEntity {id: row.entity_id, lab: 'lab5', dataset_sha256: $digest}) "
        "MERGE (c:Lab5BGEChunk {id: row.id}) SET c:Chunk "
        "SET c += row, c.lab = 'lab5', c.dataset_sha256 = $digest "
        "MERGE (c)-[:DESCRIBES]->(entity) "
        "RETURN count(c) AS count",
        rows=rows, digest=digest,
    ).single(strict=True)["count"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=ROOT / "prepared_recursive.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--index-timeout", type=int, default=300)
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "embedding_report_bge.json")
    args = parser.parse_args()
    if args.batch_size < 1 or args.index_timeout < 1:
        parser.error("--batch-size and --index-timeout must be positive")
    prepared = read_json(args.prepared)
    digest = dataset_hash(prepared)
    if prepared["metadata"].get("splitter") != "recursive_zwsp":
        raise ValueError("Run prepare_data.py --splitter recursive_zwsp --output prepared_recursive.json first.")
    chunks = prepared["chunks"]
    seen = set()
    for chunk in chunks:
        if not isinstance(chunk.get("id"), str) or not chunk["id"] or chunk["id"] in seen:
            raise ValueError("Chunk ids must be unique, nonempty strings.")
        if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
            raise ValueError(f"Chunk {chunk['id']} has no text.")
        if not isinstance(chunk.get("entity_id"), str) or not chunk["entity_id"]:
            raise ValueError(f"Chunk {chunk['id']} has no entity_id.")
        seen.add(chunk["id"])
    if not chunks:
        raise ValueError("No chunks to embed; run prepare_data.py first.")
    start = time.monotonic()
    with get_driver() as driver, driver.session(database=DATABASE) as session:
        verify_graph(session, digest, chunks)
        existing_index = index_details(session)
        if existing_index:
            verify_index(existing_index)
        session.run(
            "CREATE CONSTRAINT lab5_bge_chunk_id IF NOT EXISTS "
            "FOR (c:Lab5BGEChunk) REQUIRE c.id IS UNIQUE"
        ).consume()
        previous = {
            row["id"]: row
            for row in session.run(
                "MATCH (c:Lab5BGEChunk {lab: 'lab5', dataset_sha256: $digest}) "
                "RETURN c.id AS id, c.text AS text, c.entity_id AS entity_id, "
                "c.embedding_model AS model, c.embedding_text_sha256 AS text_sha256, "
                "size(c.embedding) AS dimensions, "
                "EXISTS { MATCH (c)-[:DESCRIBES]->"
                "(:AttackEntity {id: c.entity_id, lab: 'lab5', dataset_sha256: $digest}) } AS linked",
                digest=digest,
            ).data()
        }
        extra_ids = set(previous) - seen
        if extra_ids:
            raise ValueError(
                f"Found {len(extra_ids)} old chunks outside prepared.json. "
                "Use the original chunking settings or a separate Neo4j instance/database."
            )
        pending = []
        for chunk in chunks:
            old = previous.get(chunk["id"], {})
            if not (
                old.get("text") == chunk["text"]
                and old.get("entity_id") == chunk["entity_id"]
                and old.get("model") == EMBEDDING_MODEL
                and old.get("text_sha256") == text_hash(chunk["text"])
                and old.get("dimensions") == DIMENSIONS
                and old.get("linked")
            ):
                pending.append(chunk)
        resumed = len(chunks) - len(pending)
        print(f"Chunks: {len(chunks)}; already embedded: {resumed}; pending: {len(pending)}", flush=True)
        session.run(
            "MERGE (m:Lab5Metadata {id: 'bge_m3_recursive'}) "
            "SET m.sha256 = $digest, m.graph_complete = true, m.embeddings_complete = false, m.embedding_model = $model",
            model=EMBEDDING_MODEL, digest=digest,
        ).consume()
        if pending:
            print(f"Loading {EMBEDDING_MODEL} ...", flush=True)
            model = load_model()
            if model.get_sentence_embedding_dimension() != DIMENSIONS:
                raise ValueError(f"This lab requires {DIMENSIONS}-dimension embeddings.")
            completed = resumed
            for batch in batches(pending, args.batch_size):
                vectors = model.encode(
                    [chunk["text"] for chunk in batch],
                    batch_size=args.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                if len(vectors) != len(batch):
                    raise RuntimeError("Embedding model returned the wrong number of vectors.")
                rows = []
                for chunk, vector in zip(batch, vectors):
                    values = [float(value) for value in vector]
                    if len(values) != DIMENSIONS or not all(math.isfinite(value) for value in values):
                        raise ValueError("Embedding must contain 1024 finite floating-point values.")
                    if not any(value != 0 for value in values):
                        raise ValueError("An all-zero vector cannot be used for cosine search.")
                    rows.append({
                        "id": chunk["id"], "text": chunk["text"],
                        "entity_id": chunk["entity_id"],
                        "entity_name": chunk.get("entity_name", ""),
                        "entity_type": chunk.get("entity_type", ""),
                        "embedding": values, "embedding_model": EMBEDDING_MODEL,
                        "embedding_text_sha256": text_hash(chunk["text"]),
                    })
                count = session.execute_write(store_batch, rows, digest)
                if count != len(rows):
                    raise RuntimeError("Some chunk entities are missing; embeddings incomplete.")
                completed += count
                print(f"Embedded {completed}/{len(chunks)} chunks", flush=True)
        # Neo4j 5.16+ accepts the index name as a parameter.
        session.run(
            "CREATE VECTOR INDEX $name IF NOT EXISTS "
            "FOR (c:Lab5BGEChunk) ON c.embedding "
            "OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}}",
            name=INDEX_NAME,
        ).consume()
        deadline = time.monotonic() + args.index_timeout
        while True:
            details = index_details(session)
            verify_index(details)
            if details["state"] == "ONLINE":
                break
            if details["state"] == "FAILED":
                raise RuntimeError(f"Vector index failed: {details['failureMessage']}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Vector index is still populating; rerun create_embeddings.py to check again.")
            print(f"Vector index: {details['state']} ({details['populationPercent']:.1f}%)", flush=True)
            time.sleep(2)
        counts = session.run(
            "MATCH (c:Lab5BGEChunk {lab: 'lab5', dataset_sha256: $digest}) "
            "WHERE c.embedding_model = $model AND size(c.embedding) = $dimensions "
            "AND EXISTS { MATCH (c)-[:DESCRIBES]->"
            "(:AttackEntity {id: c.entity_id, lab: 'lab5', dataset_sha256: $digest}) } "
            "RETURN count(c) AS chunks",
            digest=digest, model=EMBEDDING_MODEL, dimensions=DIMENSIONS,
        ).single(strict=True)
        if counts["chunks"] != len(chunks):
            raise RuntimeError("Final chunk verification failed; embeddings_complete remains false.")
        session.run(
            "MATCH (m:Lab5Metadata {id: 'bge_m3_recursive'}) "
            "SET m.embeddings_complete = true, m.chunk_count = $count, "
            "m.embedding_dimensions = $dimensions, m.vector_index = $index, m.updated_at = datetime()",
            count=len(chunks), dimensions=DIMENSIONS, index=INDEX_NAME,
        ).consume()
        report = {
            "dataset_sha256": digest, "model": EMBEDDING_MODEL,
            "dimensions": DIMENSIONS, "chunks": len(chunks),
            "embedded_this_run": len(pending), "resumed_chunks": resumed,
            "index": INDEX_NAME, "index_state": details["state"],
            "index_label": "Lab5BGEChunk", "similarity": "cosine",
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }
    write_json(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved {args.report}")


if __name__ == "__main__":
    main()
