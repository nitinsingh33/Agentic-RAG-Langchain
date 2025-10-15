"""Gemini response generator service."""
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings

def get_chat_model():
    """Get configured Gemini chat model."""
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        temperature=settings.GEMINI_TEMPERATURE,
        google_api_key=settings.GEMINI_API_KEY
    )