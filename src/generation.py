from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a government welfare scheme eligibility assistant for Indian citizens.

Rules you MUST follow:
1. Answer ONLY using the information in the provided context below. Do not use outside knowledge.
2. For every claim, cite the scheme name in parentheses, e.g. (PM-KISAN).
3. If the context does not contain enough information to answer confidently, say so explicitly
   rather than guessing. Do not make up eligibility criteria, amounts, or dates.
4. Be clear and direct — the user is trying to find out what financial support they may qualify for.

Context:
{context}

Question: {question}

Answer:"""

GENERATION_PROMPT = ChatPromptTemplate.from_template(SYSTEM_PROMPT)


def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,  # low temp — we want grounded, not creative, answers
    )


def format_docs(docs):
    formatted = []
    for doc in docs:
        scheme = doc.metadata.get("scheme_name", "Unknown scheme")
        formatted.append(f"[{scheme}] {doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def generate_answer(question: str, retrieved_docs, llm):
    context = format_docs(retrieved_docs)
    chain = GENERATION_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})