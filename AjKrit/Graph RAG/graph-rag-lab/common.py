"""Shared paths and configuration for the Lab 5 scripts."""
import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
# This lab has its own login. Never infer a login from the previous student ID.
load_dotenv(ROOT / '.env')
DATABASE = os.getenv('NEO4J_DATABASE', 'neo4j')
EMBEDDING_MODEL = 'BAAI/bge-m3'
INDEX_NAME = 'lab5_bge_m3_recursive_index'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def get_driver():
    from neo4j import GraphDatabase
    password = os.getenv('NEO4J_PASSWORD')
    if not password:
        raise RuntimeError('Set NEO4J_PASSWORD in graph-rag-lab/.env first.')
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        auth=(os.getenv('NEO4J_USERNAME', 'neo4j'), password),
    )
    try:
        driver.verify_connectivity()
    except Exception:
        driver.close()
        raise
    return driver


@lru_cache(maxsize=1)
def load_model():
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = SentenceTransformer(EMBEDDING_MODEL, device=os.getenv('EMBEDDING_DEVICE', 'cpu'))
    if model.get_sentence_embedding_dimension() != 1024:
        raise RuntimeError('Expected 1024-dimensional embeddings.')
    return model
