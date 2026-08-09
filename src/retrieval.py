from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker


def build_hybrid_retriever(vectorstore, chunks, k=15):
    """Combines dense (Chroma) + sparse (BM25) retrieval via weighted ensemble."""
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.5, 0.5],  # tune later based on eval results
    )
    return hybrid_retriever


def build_reranked_retriever(hybrid_retriever, top_n=5):
    """Wraps the hybrid retriever with a cross-encoder reranker, keeping top_n."""
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)

    reranked_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid_retriever,
    )
    return reranked_retriever




if __name__ == "__main__":
    from src.ingestion import load_and_chunk_pdfs
    from src.indexing import build_vector_store

    chunks = load_and_chunk_pdfs()
    vectorstore = build_vector_store(chunks)
    hybrid = build_hybrid_retriever(vectorstore, chunks)
    reranked = build_reranked_retriever(hybrid)

    query = "I am a farmer with 2 acres of land, am I eligible for any scheme?"
    results = reranked.invoke(query)

    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ({doc.metadata.get('scheme_name')}) ---")
        print(doc.page_content[:300])