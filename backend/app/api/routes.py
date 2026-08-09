"""HTTP API for the core RAG system."""
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.config import settings
from app.rag.ingestion import ingest_documents
from app.rag.pipeline import run_rag


router = APIRouter(prefix="/api/v1")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=settings.TOP_K, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


@router.get("/")
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "ok",
    }


@router.get("/health")
async def health():
    return {"status": "ok", "service": settings.PROJECT_NAME}


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        return run_rag(request.query, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Keep provider internals out of the client response.
        raise HTTPException(status_code=500, detail="RAG query failed.") from exc


@router.post("/ingest")
async def ingest():
    try:
        return ingest_documents(settings.DATA_DIRECTORY)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Document ingestion failed.") from exc


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Save one supported document into backend/data and ingest it."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    suffix = Path(file.filename).suffix.lower()
    allowed = {".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".xls"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix or 'none'}",
        )

    settings.DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    target = settings.DATA_DIRECTORY / Path(file.filename).name
    content = await file.read()
    target.write_bytes(content)

    try:
        result = ingest_documents(settings.DATA_DIRECTORY)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Document ingestion failed.") from exc

    return {"filename": target.name, **result}
