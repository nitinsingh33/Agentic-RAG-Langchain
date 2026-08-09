"""Document ingestion pipeline: files -> chunks -> Pinecone."""

import hashlib
from pathlib import Path

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
    settings.validate()

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

    for i, chunk in enumerate(chunks):
        chunk_id = _stable_id(chunk, i)

        vector = embeddings.embed_query(
            chunk.page_content
        )

        metadata = dict(chunk.metadata)

        # Pinecone needs the actual text in metadata
        # so retrieval can reconstruct a Document.
        metadata["text"] = chunk.page_content
        metadata["chunk_id"] = i

        vectors.append(
            {
                "id": chunk_id,
                "values": vector,
                "metadata": metadata,
            }
        )

    # Upsert in batches to avoid sending a huge request.
    batch_size = 100

    for start in range(0, len(vectors), batch_size):
        batch = vectors[
            start : start + batch_size
        ]

        index.upsert(vectors=batch)

    return {
        "status": "success",
        "documents_loaded": len(unique),
        "chunks_created": len(chunks),
        "index_name": settings.INDEX_NAME,
    }