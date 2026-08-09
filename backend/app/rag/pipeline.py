"""Core RAG orchestration."""
from app.rag.generation import generate_answer
from app.rag.retrieval import retrieve


def run_rag(query: str, top_k: int | None = None) -> dict:
    documents = retrieve(query, top_k=top_k)
    answer = generate_answer(query, documents)

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
    }
