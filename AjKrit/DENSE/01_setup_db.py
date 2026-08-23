# -*- coding: utf-8 -*-

from chroma_config import (
    get_client,
    DB_PATH
)


# ============================================================
# STEP 1: INITIALIZE CHROMADB
# ============================================================

client = get_client()


print("=" * 70)
print("STEP 1: CHROMADB PERSISTENT STORAGE")
print("=" * 70)

print("ChromaDB Client initialized successfully!")
print("Database path:", DB_PATH)


# แสดง collection ที่มีอยู่
collections = client.list_collections()

print()
print("Existing collections:")

if len(collections) == 0:
    print("  No collections found.")
else:
    for collection in collections:
        print(" ", collection.name)