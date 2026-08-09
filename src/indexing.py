from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

CHROMA_DIR = "chroma_db"

def build_vector_store(chunks, persist_dir: str = CHROMA_DIR):
    """Embeds chunks and stores them in a persistent Chroma vector store."""
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"Indexed {len(chunks)} chunks into Chroma at '{persist_dir}'")
    return vectorstore


def build_bm25_index(chunks):
    """Builds an in-memory BM25 index over the same chunks (for hybrid search)."""
    tokenized_corpus = [chunk.page_content.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunks  # return chunks too, since BM25 needs the original text mapped back