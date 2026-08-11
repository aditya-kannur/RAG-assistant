from src.ingestion import load_and_chunk_pdfs
from src.indexing import build_vector_store
from src.retrieval import build_hybrid_retriever, build_reranked_retriever
from src.query_translation import generate_query_variants
from src.generation import get_llm, generate_answer
from src.query_construction import extract_scheme_filter


def build_pipeline():
    chunks = load_and_chunk_pdfs()
    vectorstore = build_vector_store(chunks)
    hybrid = build_hybrid_retriever(vectorstore, chunks)
    reranked_retriever = build_reranked_retriever(hybrid)
    llm = get_llm()
    return reranked_retriever, llm


def answer_question(question: str, vectorstore, chunks, llm):
    scheme_filter = extract_scheme_filter(question)

    hybrid = build_hybrid_retriever(vectorstore, chunks, scheme_filter=scheme_filter)
    retriever = build_reranked_retriever(hybrid)

    query_variants = generate_query_variants(question, llm)

    seen = set()
    all_docs = []
    for q in query_variants:
        docs = retriever.invoke(q)
        for doc in docs:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)

    answer = generate_answer(question, all_docs, llm)
    return answer, all_docs, scheme_filter

if __name__ == "__main__":
    retriever, llm = build_pipeline()

    query = "I am a farmer with 2 acres of land, am I eligible for any scheme?"
    answer, docs = answer_question(query, retriever, llm)

    print("\n=== ANSWER ===")
    print(answer)
    print(f"\n=== SOURCES USED ({len(docs)} chunks) ===")
    for doc in docs:
        print(f"- {doc.metadata.get('scheme_name')} (page {doc.metadata.get('page')})")