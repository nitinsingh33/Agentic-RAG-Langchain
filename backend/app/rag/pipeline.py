"""Public RAG pipeline entry point, backed by the agent manager."""

from app.agents.manager import run_agentic_rag, run_single_pass_rag


def run_rag(query: str, top_k: int | None = None, mode: str = "agentic") -> dict:
    if mode == "baseline":
        return run_single_pass_rag(query, top_k=top_k)
    return run_agentic_rag(query, top_k=top_k)
