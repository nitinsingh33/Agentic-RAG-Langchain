"""Validated HTTP request and response models."""

from typing import Any

from pydantic import BaseModel, Field

from app.core.config import settings


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=settings.TOP_K, ge=1, le=20)


class QueryMetrics(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    retrieved_chunks: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    metrics: QueryMetrics
