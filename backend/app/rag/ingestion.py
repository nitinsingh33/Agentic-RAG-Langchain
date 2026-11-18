"""Document ingestion pipeline: files -> chunks -> Pinecone."""

import hashlib
from pathlib import Path
from time import perf_counter

import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.services.retriever import get_embeddings, get_index


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".csv",
    ".xlsx",
    ".xls",
}


def _clean(text: str) -> str:
    return "\n".join(
        line.strip()
        for line in text.replace("\r", "\n").splitlines()
        if line.strip()
    )


def _row_to_document(
    row,
    columns,
    filename: str,
    source_type: str,
    row_index: int,
) -> Document | None:
    text = " | ".join(
        f"{column}: {str(row[column]).strip()}"
        for column in columns
        if str(row[column]).strip()
    )

    text = _clean(text)

    if not text:
        return None

    return Document(
        page_content=text,
        metadata={
            "filename": filename,
            "source_type": source_type,
            "row_index": row_index,
        },
    )


def _load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        docs = PyPDFLoader(str(path)).load()

    elif suffix == ".docx":
        docs = Docx2txtLoader(str(path)).load()

    elif suffix in {".txt", ".md"}:
        docs = TextLoader(
            str(path),
            encoding="utf-8",
        ).load()

    elif suffix == ".csv":
        df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
        )

        docs = [
            d
            for i, row in df.iterrows()
            if (
                d := _row_to_document(
                    row,
                    df.columns,
                    path.name,
                    "csv",
                    int(i),
                )
            )
        ]

    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(
            path,
            dtype=str,
        )

        docs = [
            d
            for i, row in df.iterrows()
            if (
                d := _row_to_document(
                    row,
                    df.columns,
                    path.name,
                    "spreadsheet",
                    int(i),
                )
            )
        ]

    else:
        return []

    for doc in docs:
        doc.page_content = _clean(doc.page_content)
        doc.metadata.setdefault("filename", path.name)
        doc.metadata.setdefault(
            "source_type",
            suffix.lstrip("."),
        )

    return [
        doc
        for doc in docs
        if doc.page_content
    ]


def _stable_id(
    doc: Document,
    chunk_number: int,
) -> str:
    raw = (
        f"{doc.metadata.get('filename', '')}"
        f"|{doc.metadata.get('page', '')}"
        f"|{chunk_number}"
        f"|{doc.page_content}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def load_documents(
    data_path: Path,
) -> list[Document]:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_path}"
        )

    if data_path.is_file():
        if data_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return []
        return _load_file(data_path)

    documents: list[Document] = []

    for path in sorted(data_path.rglob("*")):
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):
            documents.extend(_load_file(path))

    return documents


def ingest_documents(
    data_path: Path | None = None,
) -> dict:
    started_at = perf_counter()
    settings.validate_vector_store()

    data_path = data_path or settings.DATA_DIRECTORY

    raw_docs = load_documents(Path(data_path))

    if not raw_docs:
        return {
            "status": "no_documents",
            "documents_loaded": 0,
            "chunks_created": 0,
            "index_name": settings.INDEX_NAME,
        }

    # Remove exact duplicate documents.
    unique: dict[str, Document] = {}

    for doc in raw_docs:
        digest = hashlib.sha256(
            doc.page_content.encode("utf-8")
        ).hexdigest()

        unique.setdefault(digest, doc)

    # Split documents into chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(
        list(unique.values())
    )

    # Create deterministic IDs.
    vectors = []

    embeddings = get_embeddings()
    index = get_index()

    chunk_texts = [chunk.page_content for chunk in chunks]
    embedding_started_at = perf_counter()
    embedded_chunks = embeddings.embed_documents(chunk_texts)
    embedding_ms = (perf_counter() - embedding_started_at) * 1000

    for i, (chunk, values) in enumerate(zip(chunks, embedded_chunks)):
        chunk_id = _stable_id(chunk, i)

        metadata = dict(chunk.metadata)

        # Pinecone needs the actual text in metadata
        # so retrieval can reconstruct a Document.
        metadata["text"] = chunk.page_content
        metadata["chunk_id"] = i

        vectors.append(
            {
                "id": chunk_id,
                "values": values,
                "metadata": metadata,
            }
        )

    # Upsert in batches to avoid sending a huge request.
    batch_size = 100
    upsert_started_at = perf_counter()

    for start in range(0, len(vectors), batch_size):
        batch = vectors[
            start : start + batch_size
        ]

        index.upsert(vectors=batch, namespace=settings.PINECONE_NAMESPACE)
    upsert_ms = (perf_counter() - upsert_started_at) * 1000

    return {
        "status": "success",
        "documents_loaded": len(unique),
        "chunks_created": len(chunks),
        "index_name": settings.INDEX_NAME,
        "namespace": settings.PINECONE_NAMESPACE,
        "metrics": {
            "embedding_ms": round(embedding_ms, 2),
            "upsert_ms": round(upsert_ms, 2),
            "total_ms": round((perf_counter() - started_at) * 1000, 2),
        },
    }
