"""Pinecone retriever service."""
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings
import os

def get_retriever():
    """Get Pinecone retriever with HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    
    vectorstore = PineconeVectorStore(
        embedding=embeddings,
        index_name=settings.INDEX_NAME
    )
    
    return vectorstore.as_retriever()