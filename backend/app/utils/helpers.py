"""Helper functions and utilities."""
from typing import List, Dict, Any
import os
from pathlib import Path

def format_source_documents(docs: List[Any], limit: int = 5) -> List[Dict[str, Any]]:
    """Format source documents for API response."""
    formatted_sources = []
    
    for doc in docs[:limit]:
        if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
            formatted_sources.append({
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "filename": doc.metadata.get("filename", "Unknown"),
                "page": str(doc.metadata.get("page", "N/A")),
                "metadata": doc.metadata
            })
    
    return formatted_sources

def validate_file_path(file_path: str) -> bool:
    """Validate if file path exists and is readable."""
    path = Path(file_path)
    return path.exists() and path.is_file()

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent

def ensure_directory(dir_path: str) -> None:
    """Ensure directory exists, create if not."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def get_supported_file_extensions() -> List[str]:
    """Get list of supported file extensions."""
    return ['.pdf', '.docx', '.txt', '.csv', '.md']

def is_supported_file(file_path: str) -> bool:
    """Check if file has supported extension."""
    return Path(file_path).suffix.lower() in get_supported_file_extensions()