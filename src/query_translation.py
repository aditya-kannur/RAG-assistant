from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """You are helping a user find government welfare schemes they may be eligible for.
Generate 3 different rephrasings of the user's question below, using varied terminology
that might match how official scheme documents describe eligibility
(e.g. income levels, land size, occupation, age, family status).

Return ONLY the 3 rephrased questions, one per line, no numbering, no extra text.

Original question: {question}"""
)


def get_multi_query_chain(llm):
    return MULTI_QUERY_PROMPT | llm | StrOutputParser()


def generate_query_variants(question: str, llm) -> list[str]:
    chain = get_multi_query_chain(llm)
    result = chain.invoke({"question": question})
    variants = [line.strip() for line in result.split("\n") if line.strip()]
    return [question] + variants  # include original + variants