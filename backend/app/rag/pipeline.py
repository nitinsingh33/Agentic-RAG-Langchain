"""Core RAG orchestration."""
from time import perf_counter

from app.rag.generation import generate_answer
from app.rag.retrieval import retrieve


def run_rag(query: str, top_k: int | None = None) -> dict:
    started_at = perf_counter()
    retrieval_started_at = perf_counter()
    documents = retrieve(query, top_k=top_k)
    retrieval_ms = (perf_counter() - retrieval_started_at) * 1000
    generation_started_at = perf_counter()
    answer = generate_answer(query, documents)
    generation_ms = (perf_counter() - generation_started_at) * 1000

    sources = []
    for doc in documents:
        metadata = doc.metadata or {}
        sources.append(
            {
                "filename": metadata.get("filename", "unknown"),
                "page": metadata.get("page"),
                "row_index": metadata.get("row_index"),
                "source_type": metadata.get("source_type"),
                "chunk_id": metadata.get("chunk_id"),
                "content": doc.page_content[:300],
            }
        )

    return {
        "answer": answer,
        "sources": sources,
        "metrics": {
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "total_ms": round((perf_counter() - started_at) * 1000, 2),
            "retrieved_chunks": len(documents),
        },
    }
