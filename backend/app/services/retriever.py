"""Vector retrieval service using Pinecone SDK directly."""

from functools import lru_cache

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Create and cache the embedding model."""
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL
    )


@lru_cache(maxsize=1)
def get_pinecone_client() -> Pinecone:
    """Create and cache the Pinecone client."""
    settings.validate()
    return Pinecone(api_key=settings.PINECONE_API_KEY)


def ensure_index() -> None:
    """Create the Pinecone index if it does not already exist."""
    settings.validate()

    client = get_pinecone_client()
    existing_indexes = client.list_indexes().names()

    if settings.INDEX_NAME not in existing_indexes:
        client.create_index(
            name=settings.INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.PINECONE_CLOUD,
                region=settings.PINECONE_REGION,
            ),
        )


@lru_cache(maxsize=1)
def get_index():
    """Return the Pinecone index used by the application."""
    ensure_index()
    return get_pinecone_client().Index(settings.INDEX_NAME)


def get_retriever(top_k: int | None = None):
    """
    Return a simple retriever function backed by Pinecone.

    The function accepts a text query and returns LangChain Documents.
    """
    k = top_k or settings.TOP_K
    embeddings = get_embeddings()
    index = get_index()

    def retrieve(query: str) -> list[Document]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_vector = embeddings.embed_query(query)

        response = index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True,
        )

        documents = []

        for match in response.matches:
            metadata = dict(match.metadata or {})

            text = (
                metadata.pop("text", None)
                or metadata.pop("page_content", None)
                or metadata.pop("content", None)
                or ""
            )

            metadata["score"] = float(match.score)

            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        return documents

    return retrieve