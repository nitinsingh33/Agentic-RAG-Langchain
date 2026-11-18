"""Gemini generation service."""
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings


@lru_cache(maxsize=1)
def get_chat_model() -> ChatGoogleGenerativeAI:
    settings.validate_generation()
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        temperature=settings.GEMINI_TEMPERATURE,
        google_api_key=settings.GEMINI_API_KEY,
    )
