"""
Public entry point for the RAG pipeline: retrieve + generate.
"""

from src.config import TOP_K
from src.embedder import embed_query
from src.vector_store import search
from src.generator import generate_from_context


def retrieve(query: str, k: int = TOP_K) -> list[str]:
    """Embeds the query and returns the top-k matching chunk texts."""
    query_vector = embed_query(query)
    return search(query_vector, k=k)


def generate_answer(query: str, k: int = TOP_K) -> str:
    """Full pipeline: retrieve context, then generate a grounded answer."""
    context_chunks = retrieve(query, k=k)
    return generate_from_context(query, context_chunks)
