import os
import uuid
import csv
import chromadb
from pypdf import PdfReader
import openpyxl


# Phase 16 — Document Upload to RAG
# Supports: PDF, TXT, XLSX, and plain text input ("remember this:")
# Extracts text, splits into chunks, stores in ChromaDB.

# Phase 17 — Multi-Namespace RAG
# Instead of one flat "uploaded_documents" collection, documents are now
# stored in separate namespace collections based on their type.
# This lets the namespace router retrieve only the relevant org knowledge
# for a given post request, rather than mixing everything together.

# Phase 23 — Railway: use DATA_DIR env var so ChromaDB persists on Railway volume
import django.conf
DB_PATH = os.path.join(getattr(django.conf.settings, 'DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "chroma_db")

# Phase 17 — Namespace registry: human-readable key → ChromaDB collection name
NAMESPACES = {
    "company_profile":  "org_company_profile",
    "products":         "org_products",
    "brand_guidelines": "org_brand_guidelines",
    "audience":         "org_audience",
    "campaigns":        "org_campaigns",
}

_client = None

# Phase 16 — single flat collection (kept for reference)
# _docs_collection = None
# def _get_collection():
#     global _client, _docs_collection
#     if _docs_collection is None:
#         _client = chromadb.PersistentClient(path=DB_PATH)
#         _docs_collection = _client.get_or_create_collection(name="uploaded_documents")
#     return _docs_collection


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
    return _client


def _get_namespaced_collection(doc_type: str):
    """Get or create the ChromaDB collection for a given namespace key."""
    collection_name = NAMESPACES.get(doc_type, "org_company_profile")
    return _get_client().get_or_create_collection(name=collection_name)


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


# Phase 21 — CSV ingestion
# Reads CSV rows, formats as "Header: Value | Header: Value" per row
# so the LLM understands structure (column names + values together)
def extract_text_from_csv(file_path: str) -> str:
    lines = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        lines.append("Columns: " + " | ".join(headers))
        for i, row in enumerate(reader):
            row_text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
            if row_text.strip():
                lines.append(f"Row {i + 1}: {row_text}")
    return "\n".join(lines)


def extract_text(file_path: str, filename: str) -> str:
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        return extract_text_from_pdf(file_path)
    elif ext == "txt":
        return extract_text_from_txt(file_path)
    elif ext in ("xlsx", "xls"):
        return extract_text_from_excel(file_path)
    elif ext == "csv":
        return extract_text_from_csv(file_path)
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

def store_document(file_path: str, filename: str, doc_type: str = "company_profile") -> int:
    """Extract text from file, chunk it, store in the correct namespace collection."""
    text = extract_text(file_path, filename)
    if not text.strip():
        return 0
    # Phase 22 — extract company entities and save to company_config.json
    from .entity_extractor_skill import extract_and_save_entities
    extract_and_save_entities(text, doc_type)
    return _store_text(text, filename, doc_type)


def store_plain_text(text: str, label: str = "user_note", doc_type: str = "company_profile") -> int:
    """Store plain text directly — for 'remember this:' input."""
    return _store_text(text, label, doc_type)


# Phase 16 — old flat _store_text (kept for reference)
# def _store_text(text: str, filename: str) -> int:
#     chunks = chunk_text(text)
#     collection = _get_collection()
#     for i, chunk in enumerate(chunks):
#         doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
#         collection.add(
#             documents=[chunk],
#             metadatas=[{"filename": filename, "chunk": i}],
#             ids=[doc_id],
#         )
#     return len(chunks)

def _store_text(text: str, filename: str, doc_type: str = "company_profile") -> int:
    """Phase 17 — chunk text and store in the namespace-specific collection."""
    chunks = chunk_text(text)
    collection = _get_namespaced_collection(doc_type)
    for i, chunk in enumerate(chunks):
        doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
        collection.add(
            documents=[chunk],
            metadatas=[{"filename": filename, "chunk": i, "doc_type": doc_type}],
            ids=[doc_id],
        )
    return len(chunks)


# ── Retrieve ──────────────────────────────────────────────────────────────────

# Phase 16 — flat retrieve across all uploaded docs (kept for reference)
# def retrieve_from_documents(query: str, n_results: int = 4) -> str:
#     collection = _get_collection()
#     total = len(collection.get()["ids"])
#     if total == 0:
#         return ""
#     results = collection.query(query_texts=[query], n_results=min(n_results, total))
#     chunks = results["documents"][0]
#     metadatas = results["metadatas"][0]
#     if not chunks:
#         return ""
#     context = "Relevant content from your knowledge base:\n\n"
#     for meta, chunk in zip(metadatas, chunks):
#         context += f"From {meta['filename']}:\n{chunk}\n\n"
#     return context.strip()


def retrieve_from_namespace(query: str, namespaces: list, n_results: int = 3) -> str:
    """Phase 17 — query only the specified namespace collections, return merged context."""
    client = _get_client()
    context_parts = []

    for ns_key in namespaces:
        collection_name = NAMESPACES.get(ns_key)
        if not collection_name:
            continue

        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            continue  # namespace exists in registry but has no documents yet

        total = len(collection.get()["ids"])
        if total == 0:
            continue

        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, total),
        )
        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]

        if chunks:
            label = ns_key.replace("_", " ").title()
            part = f"[{label}]:\n"
            for meta, chunk in zip(metadatas, chunks):
                part += f"{chunk}\n\n"
            context_parts.append(part.strip())

    if not context_parts:
        return ""

    return "Organizational Knowledge:\n\n" + "\n\n".join(context_parts)


def list_documents_by_namespace() -> dict:
    """Returns {namespace_key: [filenames]} for all namespaces that have documents."""
    client = _get_client()
    result = {}
    for ns_key, collection_name in NAMESPACES.items():
        try:
            collection = client.get_collection(name=collection_name)
            metadatas = collection.get()["metadatas"]
            filenames = list({m["filename"] for m in metadatas if m})
            if filenames:
                result[ns_key] = filenames
        except Exception:
            pass
    return result


# Phase 16 — kept for reference
# def list_documents() -> list:
#     all_meta = _get_collection().get()["metadatas"]
#     return list({m["filename"] for m in all_meta if m})
