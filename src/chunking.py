"""
Splits document text into retrieval-sized chunks and builds
parallel metadata for each chunk.
"""

import pandas as pd

from src.config import CHUNK_MAX_WORDS


def chunk_text(text: str, max_words: int = CHUNK_MAX_WORDS) -> list[str]:
    """Split text into word-count-bounded chunks."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def build_chunks(df: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """
    Returns (chunks, chunk_metadata) — parallel lists where
    chunk_metadata[i] describes chunks[i].
    """
    chunks: list[str] = []
    chunk_metadata: list[dict] = []

    for idx, row in df.iterrows():
        for c in chunk_text(row["response"]):
            chunks.append(c)
            chunk_metadata.append({
                "source_id": int(idx),
                "intent": row["intent"],
                "category": row["category"],
            })

    return chunks, chunk_metadata
