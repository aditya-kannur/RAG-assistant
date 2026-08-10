from src.ingestion import load_and_chunk_pdfs
from src.indexing import build_vector_store
from src.retrieval import build_hybrid_retriever, build_reranked_retriever
from src.query_translation import generate_query_variants
from src.generation import get_llm, generate_answer


def build_pipeline():
    chunks = load_and_chunk_pdfs()
    vectorstore = build_vector_store(chunks)
    hybrid = build_hybrid_retriever(vectorstore, chunks)
    reranked_retriever = build_reranked_retriever(hybrid)
    llm = get_llm()
    return reranked_retriever, llm


def answer_question(question: str, retriever, llm):
    # 1. Multi-query: expand the question into variants
    query_variants = generate_query_variants(question, llm)

    # 2. Retrieve for each variant, dedupe by content
    seen = set()
    all_docs = []
    for q in query_variants:
        docs = retriever.invoke(q)
        for doc in docs:
            key = doc.page_content[:200]  # dedupe key
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)

    # 3. Generate answer from deduped combined context
    answer = generate_answer(question, all_docs, llm)
    return answer, all_docs


if __name__ == "__main__":
    retriever, llm = build_pipeline()

    query = "I am a farmer with 2 acres of land, am I eligible for any scheme?"
    answer, docs = answer_question(query, retriever, llm)

    print("\n=== ANSWER ===")
    print(answer)
    print(f"\n=== SOURCES USED ({len(docs)} chunks) ===")
    for doc in docs:
        print(f"- {doc.metadata.get('scheme_name')} (page {doc.metadata.get('page')})")