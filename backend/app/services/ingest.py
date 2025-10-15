import os
import hashlib
from pathlib import Path
import pandas as pd
import pinecone
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


# ---------------- CONFIG ---------------- #
load_dotenv()
DATA_PATH = Path(__file__).parent.parent.parent.parent / "data"  # Project root data folder


INDEX_NAME = os.getenv("INDEX_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

SUPPORTED_EXT = {".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".xls"}
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------- HELPERS ---------------- #
def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(s: str) -> str:
    """Remove empty lines + normalize spacing"""
    if not s:
        return ""
    return "\n".join(l.strip() for l in s.replace("\r", "\n").splitlines() if l.strip())


def csv_row_to_text(row, cols):
    """Convert dataframe row to string"""
    return " | ".join(f"{c}: {str(row[c]).strip()}" for c in cols if str(row[c]).strip())


def ensure_index_exists(dimension=384):
    """Ensure Pinecone index is available"""
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    if INDEX_NAME not in pinecone.list_indexes():
        print(f"[INFO] Creating Pinecone index '{INDEX_NAME}' with dim={dimension}...")
        pinecone.create_index(INDEX_NAME, dimension=dimension)
    else:
        print(f"[INFO] Pinecone index '{INDEX_NAME}' exists.")


# ---------------- FILE HANDLERS ---------------- #
def load_pdf(fp: Path):
    return PyPDFLoader(str(fp)).load()

def load_docx(fp: Path):
    return Docx2txtLoader(str(fp)).load()

def load_txt(fp: Path):
    return TextLoader(str(fp), encoding="utf8").load()

def load_md(fp: Path):
    """Load Markdown files using TextLoader"""
    return TextLoader(str(fp), encoding="utf8").load()

def load_csv(fp: Path):
    try:
        df = pd.read_csv(fp, dtype=str, keep_default_na=False, encoding="utf-8")
    except Exception as e:
        print(f"[ERR] reading csv {fp}: {e}")
        return []
    return [
        Document(page_content=clean_text(csv_row_to_text(row, df.columns)),
                 metadata={"filename": fp.name, "source_type": "csv", "row_index": int(idx)})
        for idx, row in df.iterrows()
        if clean_text(csv_row_to_text(row, df.columns))
    ]

def load_excel(fp: Path):
    try:
        df = pd.read_excel(fp, dtype=str)
    except Exception as e:
        print(f"[ERR] reading excel {fp}: {e}")
        return []
    return [
        Document(page_content=clean_text(csv_row_to_text(row, df.columns)),
                 metadata={"filename": fp.name, "source_type": "xlsx", "row_index": int(idx)})
        for idx, row in df.iterrows()
        if clean_text(csv_row_to_text(row, df.columns))
    ]


FILE_LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
    ".md": load_md,
    ".csv": load_csv,
    ".xlsx": load_excel,
    ".xls": load_excel,
}


# ---------------- INGESTION PIPELINE ---------------- #
def ingest_files(data_path=DATA_PATH):
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"No data folder at {data_path}")

    try:
        ensure_index_exists(dimension=384)
    except Exception as e:
        print("[WARN] Could not verify/create Pinecone index:", e)

    all_docs = []

    for fp in sorted(p.rglob("*")):
        if not fp.is_file() or fp.suffix.lower() not in SUPPORTED_EXT:
            continue

        print(f"📂 Loading {fp.name} ...")
        docs = FILE_LOADERS.get(fp.suffix.lower(), lambda x: [])(fp)

        # filter + add metadata
        for d in docs:
            if d.page_content.strip():
                d.metadata.update({"filename": fp.name, "source_type": fp.suffix.lower().lstrip(".")})
                all_docs.append(d)

    if not all_docs:
        print("ℹ️ No documents found to ingest.")
        return

    # Deduplicate
    unique_docs, seen = [], set()
    for d in all_docs:
        h = sha256(d.page_content)
        if h not in seen:
            seen.add(h)
            d.metadata["doc_hash"] = h
            unique_docs.append(d)

    print(f"✳️ Unique documents: {len(unique_docs)}")

    # Split into chunks
    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = splitter.split_documents(unique_docs)
    for i, t in enumerate(texts):
        t.metadata["chunk_id"] = i

    print(f"✂️ Created {len(texts)} chunks")

    # Embed + Push
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    PineconeVectorStore.from_documents(texts, embeddings, index_name=INDEX_NAME)

    print("✅ Ingestion complete — documents embedded to Pinecone.")
    return {
        "total_docs": len(unique_docs),
        "chunks_created": len(texts),
        "pinecone_index": INDEX_NAME
    }


if __name__ == "__main__":
    ingest_files()
