"""
Step 3: Vector Store Setup
----------------------------
Reads the knowledge base markdown files, splits them into chunks, embeds
them locally (no API call, no cost — sentence-transformers runs on your
machine), and stores them in a persistent ChromaDB collection on disk.

This is a one-time build step. The RAG Agent (Step 4/5) will query this
store, not rebuild it on every run. Re-run this script only if you add or
edit knowledge base files.
"""

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

KB_DIR = Path(__file__).parent / "data" / "knowledge_base"
CHROMA_DIR = Path(__file__).parent / "data" / "chroma_store"

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += para + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            current = para + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks


def build_store():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection("rossmann_knowledge")
    except Exception:
        pass

    collection = client.create_collection(
        name="rossmann_knowledge",
        embedding_function=embedding_fn,
    )

    all_chunks = []
    all_ids = []
    all_metadata = []

    md_files = list(KB_DIR.glob("*.md"))
    print(f"Found {len(md_files)} knowledge base files: {[f.name for f in md_files]}")

    for file_path in md_files:
        text = file_path.read_text()
        chunks = chunk_text(text)
        print(f"  {file_path.name} -> {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{file_path.stem}_{i}")
            all_metadata.append({"source": file_path.name})

    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metadata)
    print(f"Stored {len(all_chunks)} chunks in ChromaDB at {CHROMA_DIR}")

    test_query = "data leakage rolling average"
    results = collection.query(query_texts=[test_query], n_results=2)
    print(f"\nSanity check — query: '{test_query}'")
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"  [{meta['source']}] {doc[:120]}...")


if __name__ == "__main__":
    build_store()