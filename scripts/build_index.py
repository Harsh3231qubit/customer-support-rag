"""
Run this once to build the document store:
    python -m scripts.build_index

Loads the dataset, cleans it, chunks it, embeds it with MiniLM,
and uploads everything into Qdrant. Also saves chunks/embeddings
to disk so later runs can skip re-embedding (see src/config.py
for paths — point BASE_DIR at Google Drive in Colab to persist
across sessions).
"""

import os
import pickle
import numpy as np

from src.config import BASE_DIR, CHUNKS_PATH, EMBEDDINGS_PATH
from src.data_loader import load_and_clean
from src.chunking import build_chunks
from src.embedder import embed_texts
from src.vector_store import build_collection


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    print("Loading and cleaning dataset...")
    df = load_and_clean()

    print("Chunking...")
    chunks, chunk_metadata = build_chunks(df)
    print(f"  {len(chunks)} chunks created")

    print("Embedding with MiniLM...")
    embeddings = embed_texts(chunks)

    print("Saving chunks and embeddings to disk...")
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "metadata": chunk_metadata}, f)
    np.save(EMBEDDINGS_PATH, embeddings)

    print("Building Qdrant collection...")
    build_collection(chunks, chunk_metadata, embeddings)

    print("Done. Index is ready.")


if __name__ == "__main__":
    main()
