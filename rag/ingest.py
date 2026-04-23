"""
rag/ingest.py
-------------
Ingests CPChem maintenance documents into ChromaDB vector store.
Uses Nvidia NIM embeddings (free tier via build.nvidia.com).

WHY RAG FOR CARIS:
  The problem at Cedar Bayou is not that the answers don't exist.
  They do — in maintenance manuals, SOPs, and failure reports.
  The problem is a reliability engineer can't search 500 pages
  of documents in the 10 minutes they have when an alarm fires.

  RAG solves this by:
  1. Pre-indexing all documents into a vector store
  2. At query time, finding the 3-5 most relevant chunks instantly
  3. Feeding only those chunks to the LLM — no hallucination
  4. Returning the answer with source document + page reference

WHY NVIDIA NIM FOR EMBEDDINGS:
  nvidia/nv-embedqa-e5-v5 is purpose-built for RAG retrieval
  on technical/industrial documents. It outperforms
  OpenAI text-embedding-3-small on domain-specific Q&A tasks.
  It is free at 40 RPM on build.nvidia.com.

Usage:
  python -m rag.ingest
  python -m rag.ingest --docs-dir rag/docs --db-dir chroma_db
"""

import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = "rag/docs"
DB_DIR   = "chroma_db"
COLLECTION_NAME = "caris_cpchem_knowledge"

# Nvidia NIM embedding model — purpose-built for RAG
NVIDIA_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"

# chunk settings
CHUNK_SIZE    = 800   # characters per chunk
CHUNK_OVERLAP = 150   # overlap between chunks to preserve context


def check_api_key():
    key = os.getenv("NVIDIA_API_KEY", "")
    if not key.startswith("nvapi-"):
        print("ERROR: NVIDIA_API_KEY not set or invalid.")
        print("Get your free key from: https://build.nvidia.com")
        print("Add to .env file: NVIDIA_API_KEY=nvapi-xxxx")
        sys.exit(1)
    print(f"Nvidia API key found: nvapi-...{key[-6:]}")


def load_documents(docs_dir: str) -> list[dict]:
    """
    Load all .txt files from docs directory.
    Returns list of dicts with text and metadata.
    """
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    docs = []
    txt_files = [f for f in os.listdir(docs_dir) if f.endswith(".txt")]

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {docs_dir}")

    print(f"Found {len(txt_files)} documents in {docs_dir}")

    for filename in sorted(txt_files):
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({
            "text":     text,
            "filename": filename,
            "filepath": filepath,
            "chars":    len(text),
        })
        print(f"  Loaded {filename}: {len(text):,} characters")

    return docs


def chunk_document(doc: dict, chunk_size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split document text into overlapping chunks.

    WHY CHUNKING:
      LLMs have context limits. Sending the full 500-page manual
      to the LLM is impossible and expensive. We split into
      small chunks, embed each one, and only send the TOP 3-5
      most relevant chunks at query time.

    WHY OVERLAP:
      A sentence split across two chunks would be missed.
      Overlap ensures no information is lost at boundaries.
    """
    text = doc["text"]
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = start + chunk_size

        # extend to end of sentence if possible
        if end < len(text):
            period_pos = text.rfind(".", start, end + 100)
            newline_pos = text.rfind("\n", start, end + 50)
            boundary = max(period_pos, newline_pos)
            if boundary > start + chunk_size // 2:
                end = boundary + 1

        chunk_text = text[start:end].strip()

        if len(chunk_text) > 50:  # skip tiny chunks
            chunks.append({
                "text":       chunk_text,
                "filename":   doc["filename"],
                "chunk_idx":  chunk_idx,
                "chunk_id":   f"{doc['filename']}::chunk_{chunk_idx}",
                "char_start": start,
                "char_end":   end,
            })
            chunk_idx += 1

        start = end - overlap

    return chunks


def build_vector_store(chunks: list[dict], db_dir: str) -> None:
    """
    Embed all chunks using Nvidia NIM and store in ChromaDB.

    ChromaDB is a local open-source vector database.
    In production at CPChem/Frame, this would be replaced by
    Azure AI Search — same interface, enterprise scale.
    """
    try:
        import chromadb
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install chromadb langchain-nvidia-ai-endpoints langchain-community langchain")
        sys.exit(1)

    print(f"\nInitializing Nvidia NIM embeddings ({NVIDIA_EMBED_MODEL})...")
    embeddings = NVIDIAEmbeddings(
        model=NVIDIA_EMBED_MODEL,
        api_key=os.getenv("NVIDIA_API_KEY"),
        truncate="NONE",
    )

    # convert chunks to LangChain Document objects
    documents = []
    for chunk in chunks:
        documents.append(Document(
            page_content=chunk["text"],
            metadata={
                "source":    chunk["filename"],
                "chunk_id":  chunk["chunk_id"],
                "chunk_idx": chunk["chunk_idx"],
            }
        ))

    print(f"Embedding {len(documents)} chunks into ChromaDB...")
    print("(This may take 1-2 minutes on free Nvidia NIM tier)")

    # create or overwrite vector store
    if os.path.exists(db_dir):
        import shutil
        shutil.rmtree(db_dir)
        print(f"Cleared existing ChromaDB at {db_dir}")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_dir,
        collection_name=COLLECTION_NAME,
    )

    print(f"\nVector store built successfully.")
    print(f"  Documents embedded: {len(documents)}")
    print(f"  Stored at: {db_dir}/")
    print(f"  Collection: {COLLECTION_NAME}")
    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="CARIS RAG document ingestion")
    parser.add_argument("--docs-dir", default=DOCS_DIR)
    parser.add_argument("--db-dir",   default=DB_DIR)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    print("=== CARIS RAG Ingestion Pipeline ===\n")
    check_api_key()

    # load docs
    docs = load_documents(args.docs_dir)

    # chunk all docs
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, args.chunk_size)
        all_chunks.extend(chunks)
        print(f"  {doc['filename']}: {len(chunks)} chunks")

    print(f"\nTotal chunks to embed: {len(all_chunks)}")

    # embed and store
    build_vector_store(all_chunks, args.db_dir)

    print("\nIngestion complete. Run retriever next:")
    print("  python -m rag.retriever")


if __name__ == "__main__":
    main()
