"""API routes for the application."""
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import tempfile
import os

logger = logging.getLogger(__name__)

# Direct imports for production
from app.services.ingest import ingest_files   
from app.services.multimodal_processor import MultiModalProcessor, process_multimodal_document

def route_query_production(query: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """Production query routing."""
    try:
        from app.services.stateful_bot import get_qa
        result = get_qa().invoke({"question": query, "chat_history": chat_history or []})
        
        return {
            "answer": result.get("answer", str(result)),
            "intent": "general",
            "source_documents": result.get("source_documents", []),
            "agent_used": "RAG Agent"
        }
            
    except Exception as e:
        logger.error(f"Error in routing: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "answer": "I encountered an error processing your query. Please try again.",
            "intent": "error",
            "source_documents": [],
            "agent_used": "Error Handler"
        }


router = APIRouter(prefix="/api/v1")


# --------------------
# Models
# --------------------
class QueryRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] = []


class QueryResponse(BaseModel):
    answer: str
    intent: str = "auto-detected"
    source_documents: List[Dict[str, Any]] = []
    agent_used: str = "Central Router"


class IngestRequest(BaseModel):
    directory_path: str = "data"


class HealthResponse(BaseModel):
    status: str
    version: str
    backend: Dict[str, str]


class MultiModalUploadResponse(BaseModel):
    success: bool
    document_id: str
    message: str
    processing_stats: Dict[str, Any] = {}
    error: Optional[str] = None


class MultiModalQueryRequest(BaseModel):
    query: str
    query_type: str = "auto"  # auto, visual, tabular, financial, technical
    top_k: int = 5


class MultiModalQueryResponse(BaseModel):
    success: bool
    query: str
    query_type: str
    results: List[Dict[str, Any]]
    total_results: int
    error: Optional[str] = None


# --------------------
# Routes
# --------------------
@router.get("/", tags=["Root"])
async def root():
    return {
        "message": "RAG API - Backend Ready (Multi-Modal Enhanced)",
        "status": "healthy",
        "version": "1.0.0",
        "features": ["text_rag", "multi_modal", "image_analysis", "table_extraction"],
        "docs": "/docs"
    }


# --------------------
# Multi-Modal Routes
# --------------------
@router.post("/documents/multimodal", tags=["Multi-Modal"], response_model=MultiModalUploadResponse)
async def upload_multimodal_document(file: UploadFile = File(...)):
    """
    Upload and process a document with multi-modal capabilities (text + images + tables).
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.docx')):
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process with multi-modal capabilities
            result = process_multimodal_document(temp_file_path)
            
            if result.get('success', False):
                return MultiModalUploadResponse(
                    success=True,
                    document_id=result['document_id'],
                    message=f"Document '{file.filename}' processed successfully with multi-modal analysis",
                    processing_stats=result.get('processing_stats', {})
                )
            else:
                return MultiModalUploadResponse(
                    success=False,
                    document_id="",
                    message="Document processing failed",
                    error=result.get('error', 'Unknown error')
                )
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        logger.exception("❌ Multi-modal document upload failed")
        return MultiModalUploadResponse(
            success=False,
            document_id="",
            message="Document upload and processing failed",
            error=str(e)
        )


@router.post("/query/multimodal", tags=["Multi-Modal"], response_model=MultiModalQueryResponse)
async def query_multimodal_content(request: MultiModalQueryRequest):
    """
    Query multi-modal content with enhanced visual understanding capabilities.
    """
    try:
        processor = MultiModalProcessor()
        result = processor.query_multimodal_content(
            query=request.query,
            top_k=request.top_k
        )
        
        if result.get('success', False):
            return MultiModalQueryResponse(
                success=True,
                query=request.query,
                query_type=result.get('query_type', 'general'),
                results=result.get('results', []),
                total_results=result.get('total_results', 0)
            )
        else:
            return MultiModalQueryResponse(
                success=False,
                query=request.query,
                query_type='error',
                results=[],
                total_results=0,
                error=result.get('error', 'Query processing failed')
            )
    
    except Exception as e:
        logger.exception("❌ Multi-modal query failed")
        return MultiModalQueryResponse(
            success=False,
            query=request.query,
            query_type='error',
            results=[],
            total_results=0,
            error=str(e)
        )


@router.get("/multimodal/capabilities", tags=["Multi-Modal"])
async def get_multimodal_capabilities():
    """
    Get information about multi-modal capabilities.
    """
    return {
        "features": {
            "image_extraction": "Extract images from PDF documents",
            "vision_analysis": "Analyze images with Gemini Vision + OCR",
            "table_extraction": "Extract and analyze data tables",
            "chart_analysis": "Understand charts, graphs, and data visualizations",
            "technical_diagrams": "Analyze technical diagrams and flowcharts",
            "financial_analysis": "Specialized analysis for financial documents",
            "multi_modal_search": "Search across text, images, and tables",
            "ocr_text_extraction": "Extract text from images using Tesseract OCR"
        },
        "supported_formats": ["PDF", "DOCX"],
        "vision_models": ["gemini-1.5-flash (with OCR enhancement)"],
        "extraction_tools": ["PyMuPDF", "Tabula", "Camelot", "Tesseract OCR"],
        "status": "enabled",
        "note": "Using Gemini + OCR for cost-effective visual analysis"
    }


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        backend={
            "router": "router_agent.py",
            "agents": "company, pricing, sales, multimodal",
            "base_agent": "stateful_bot.py",
            "ingestion": "ingest.py"
        }
    )


def convert_chat_history_to_tuples(chat_history):
    
    if not chat_history:
        return []
    
    converted = []
    for entry in chat_history:
        if isinstance(entry, dict) and "human" in entry and "ai" in entry:
            converted.append((entry["human"], entry["ai"]))
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            converted.append((entry[0], entry[1]))
    return converted

@router.post("/query", response_model=QueryResponse, tags=["Query"])
async def process_query(request: QueryRequest):
    try:
        
        converted_history = convert_chat_history_to_tuples(request.chat_history)
        result = route_query_production(request.query, converted_history)
        answer = result.get("answer", "No answer available")
        source_docs = result.get("source_documents", [])

        formatted_sources = [
            {
                "content": (
                    doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200 else doc.page_content
                ),
                "metadata": {
                    "filename": doc.metadata.get("filename"),
                    "source_type": doc.metadata.get("source_type"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                }
            }
            for doc in source_docs[:5]
        ]

        return QueryResponse(
            answer=answer,
            intent="auto-detected",
            source_documents=formatted_sources,
            agent_used="RAG Agent"
        )

    except Exception as e:
        logger.exception("❌ Query processing failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ingest", tags=["Ingestion"])
async def ingest_documents(request: IngestRequest):
    try:
        result = ingest_files(request.directory_path)  # should return dict summary
        return {
            "message": f"Documents ingested successfully from {request.directory_path}",
            "status": "success",
            "result": result
        }
    except Exception as e:
        logger.exception("❌ Document ingestion failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/agents", tags=["Agents"])
async def get_agents():
    return {
        "available_agents": ["sales", "pricing", "company", "multimodal", "general"],
        "agent_files": {
            "sales": "backend/app/agents/sales_agent.py",
            "pricing": "backend/app/agents/pricing_agent.py",
            "company": "backend/app/agents/company_agent.py",
            "multimodal": "backend/app/agents/multimodal_agent.py",
            "base": "backend/app/services/stateful_bot.py"
        },
        "router": "backend/app/agents/router_agent.py with intelligent intent detection",
        "ingestion": "backend/app/services/ingest.py for document processing",
        "multi_modal": {
            "processor": "backend/app/services/multimodal_processor.py",
            "vision_service": "backend/app/services/vision_service.py",
            "image_extractor": "backend/app/utils/image_extractor.py"
        }
    }


@router.get("/status", tags=["Status"])
async def backend_status():
    return {
        "backend_structure": {
            "agents_folder": "✅ Located at backend/app/agents/",
            "base_agent": "✅ stateful_bot.py in backend/app/services/",
            "router": "✅ router.py with intent detection",
            "ingestion": "✅ ingest.py for documents",
            "main_api": "✅ FastAPI application"
        },
        "structure_status": "✅ Reorganized according to recommended structure",
        "service_integration": "🔄 Pending - after structure completion"
    }
