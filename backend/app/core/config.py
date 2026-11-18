"""Application configuration.

All secrets are loaded from one file: backend/.env
Never commit that file. Use backend/.env.example as the template.
"""
from pathlib import Path
import os
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BACKEND_DIR / ".env"
load_dotenv(ENV_FILE)


class Settings:
    PROJECT_NAME = os.getenv("PROJECT_NAME", "EV Moto GPT API")
    VERSION = os.getenv("VERSION", "1.0.0")
    DESCRIPTION = os.getenv(
        "DESCRIPTION",
        "Agentic document research API for Hero Motocorp business knowledge.",
    )

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = os.getenv("INDEX_NAME", "ev-gpt-rag")
    PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
    PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")
    PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "384"))

    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0"))

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
    TOP_K = int(os.getenv("TOP_K", "5"))
    MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(25 * 1024 * 1024)))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8001"))
    RELOAD = os.getenv("RELOAD", "false").lower() == "true"

    DATA_DIRECTORY = BACKEND_DIR / os.getenv("DATA_DIRECTORY", "data")

    ALLOWED_ORIGINS = [
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")
        if origin.strip()
    ]

    def validate_vector_store(self) -> None:
        missing = []
        if not self.PINECONE_API_KEY:
            missing.append("PINECONE_API_KEY")
        if not self.INDEX_NAME:
            missing.append("INDEX_NAME")

        if missing:
            raise RuntimeError(
                "Missing required environment variables: " + ", ".join(missing)
            )

    def validate_generation(self) -> None:
        if not self.GEMINI_API_KEY:
            raise RuntimeError("Missing required environment variable: GEMINI_API_KEY")

    def validate(self) -> None:
        self.validate_vector_store()
        self.validate_generation()


settings = Settings()
