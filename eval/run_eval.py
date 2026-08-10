import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from src.ingestion import load_and_chunk_pdfs
from src.indexing import build_vector_store
from src.generation import get_llm
from src.pipeline import answer_question

from ragas.run_config import RunConfig


from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

def run_evaluation():
    chunks = load_and_chunk_pdfs()
    vectorstore = build_vector_store(chunks)
    llm = get_llm()

    # Wrap Groq LLM and local embeddings for RAGAS to use internally
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )

    with open("eval/test_questions.json") as f:
        test_set = json.load(f)

    questions, ground_truths, answers, contexts = [], [], [], []

    for item in test_set:
        answer, docs, _ = answer_question(item["question"], vectorstore, chunks, llm)
        questions.append(item["question"])
        ground_truths.append(item["ground_truth"])
        answers.append(answer)
        contexts.append([doc.page_content for doc in docs])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=RunConfig(max_workers=1, timeout=120),  # sequential, longer timeout
    )
    print(results)
    return results


if __name__ == "__main__":
    run_evaluation()