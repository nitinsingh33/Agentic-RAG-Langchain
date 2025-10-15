import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    
    # API Configuration
    PROJECT_NAME: str = "RAG API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "RAG system using FastAPI, Langchain, Pinecone, and Gemini"
    
    # API Keys
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    
    # Pinecone Configuration
    INDEX_NAME: Optional[str] = os.getenv("INDEX_NAME", "rag-index")
    PINECONE_ENVIRONMENT: Optional[str] = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    
    # Multi-Modal Configuration (Using Gemini Vision)
    multimodal_enabled: bool = os.getenv("MULTIMODAL_ENABLED", "true").lower() == "true"
    vision_model: str = os.getenv("VISION_MODEL", "gemini-1.5-pro")  # Updated model name
    max_image_size: int = int(os.getenv("MAX_IMAGE_SIZE", "2097152"))  # 2MB
    ocr_enabled: bool = os.getenv("OCR_ENABLED", "true").lower() == "true"
    table_extraction_enabled: bool = os.getenv("TABLE_EXTRACTION_ENABLED", "true").lower() == "true"
    
    # HuggingFace Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Gemini Configuration
    GEMINI_MODEL: str = "gemini-1.5-flash"
    GEMINI_TEMPERATURE: float = 0.0
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    RELOAD: bool = True
    
    # CORS Configuration
    ALLOWED_ORIGINS: list[str] = ["*"]
    ALLOWED_METHODS: list[str] = ["*"]
    ALLOWED_HEADERS: list[str] = ["*"]

    # Data Configuration
    DATA_DIRECTORY: str = "data"
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    def validate_required_keys(self) -> None:
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        if not self.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        if self.multimodal_enabled and not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required for multi-modal features (using Gemini Vision)")

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
