import os
import uuid
import chromadb
from pypdf import PdfReader
import openpyxl


# Phase 16 — Document Upload to RAG
# Supports: PDF, TXT, XLSX, and plain text input ("remember this:")
# Extracts text, splits into chunks, stores in ChromaDB.

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "chroma_db")

client = chromadb.PersistentClient(path=DB_PATH)
docs_collection = client.get_or_create_collection(name="uploaded_documents")


# ── Text extraction per file type ─────────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text_from_excel(file_path: str) -> str:
    wb = openpyxl.load_workbook(file_path, data_only=True)
    lines = []
    for sheet in wb.worksheets:
        lines.append(f"Sheet: {sheet.title}")
        for row in sheet.iter_rows(values_only=True):
            row_text = " | ".join(str(c) for c in row if c is not None)
            if row_text.strip():
                lines.append(row_text)
    return "\n".join(lines)


def extract_text(file_path: str, filename: str) -> str:
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        return extract_text_from_pdf(file_path)
    elif ext == "txt":
        return extract_text_from_txt(file_path)
    elif ext in ("xlsx", "xls"):
        return extract_text_from_excel(file_path)
    return ""


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500) -> list:
    words = text.split()
    chunks = []
    step = chunk_size - 50  # 50-word overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ── Store ─────────────────────────────────────────────────────────────────────

def store_document(file_path: str, filename: str) -> int:
    """Extract text from file, chunk it, store in ChromaDB. Returns chunk count."""
    text = extract_text(file_path, filename)
    if not text.strip():
        return 0
    return _store_text(text, filename)


def store_plain_text(text: str, label: str = "user_note") -> int:
    """Store plain text directly — for 'remember this:' input."""
    return _store_text(text, label)


def _store_text(text: str, filename: str) -> int:
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
        docs_collection.add(
            documents=[chunk],
            metadatas=[{"filename": filename, "chunk": i}],
            ids=[doc_id],
        )
    return len(chunks)


# ── Retrieve ──────────────────────────────────────────────────────────────────

def retrieve_from_documents(query: str, n_results: int = 2) -> str:
    total = len(docs_collection.get()["ids"])
    if total == 0:
        return ""

    results = docs_collection.query(
        query_texts=[query],
        n_results=min(n_results, total),
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not chunks:
        return ""

    context = "Relevant content from your knowledge base:\n\n"
    for meta, chunk in zip(metadatas, chunks):
        context += f"From {meta['filename']}:\n{chunk}\n\n"

    return context.strip()


def list_documents() -> list:
    all_meta = docs_collection.get()["metadatas"]
    return list({m["filename"] for m in all_meta if m})
