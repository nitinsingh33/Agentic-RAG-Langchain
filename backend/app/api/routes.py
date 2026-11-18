"""HTTP API for the core RAG system."""
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.api.schemas import QueryRequest, QueryResponse
from app.rag.ingestion import ingest_documents
from app.rag.pipeline import run_rag


router = APIRouter(prefix="/api/v1")


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
        return await run_in_threadpool(run_rag, request.query, request.top_k, request.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Keep provider internals out of the client response.
        raise HTTPException(status_code=500, detail="RAG query failed.") from exc


@router.post("/ingest")
def ingest():
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
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the size limit.")
    target.write_bytes(content)

    try:
        result = await run_in_threadpool(ingest_documents, target)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Document ingestion failed.") from exc

    return {"filename": target.name, **result}
