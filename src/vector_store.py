"""
Thin wrapper around Qdrant for building and querying the support_kb
collection. Uses local (embedded) mode by default — no server required.
"""

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from src.config import QDRANT_PATH, COLLECTION_NAME, EMBED_DIM

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Lazily connects to the local/persisted Qdrant instance."""
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_PATH)
    return _client


def build_collection(chunks: list[str], chunk_metadata: list[dict], embeddings: np.ndarray, batch_size: int = 512) -> None:
    """
    Creates (or recreates) the collection and uploads all points.
    WARNING: recreate_collection wipes any existing data under this name.
    """
    client = get_client()

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={"text": chunks[i], **chunk_metadata[i]},
        )
        for i in range(len(chunks))
    ]

    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i:i + batch_size])


def search(query_vector: list[float], k: int = 5) -> list[str]:
    """Returns the top-k retrieved chunk texts for a query vector."""
    client = get_client()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
    )
    return [point.payload["text"] for point in results.points]
