"""Retrieval layer for the core RAG pipeline."""
from app.services.retriever import get_retriever


def retrieve(query: str, top_k: int | None = None):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    return get_retriever(top_k)(query)
