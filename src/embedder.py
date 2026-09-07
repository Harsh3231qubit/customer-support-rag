"""
Wraps the MiniLM sentence-transformer used for both indexing and
query-time embedding.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL_NAME

_model: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """Lazily loads the embedding model once per process."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    """Embed a list of texts into normalized vectors (for cosine similarity)."""
    model = get_embedder()
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    model = get_embedder()
    return model.encode([query], normalize_embeddings=True)[0].tolist()
