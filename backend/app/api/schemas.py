"""Validated HTTP request and response models."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.config import settings


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=settings.TOP_K, ge=1, le=20)
    mode: Literal["agentic", "baseline"] = "agentic"


class QueryMetrics(BaseModel):
    planning_ms: float
    retrieval_ms: float
    agent_ms: float
    synthesis_ms: float
    total_ms: float
    retrieved_chunks: int


class AgentStep(BaseModel):
    name: str
    role: str
    status: str
    output: str


class QueryResponse(BaseModel):
    mode: Literal["agentic", "baseline"]
    answer: str
    sources: list[dict[str, Any]]
    agents: list[AgentStep]
    metrics: QueryMetrics
