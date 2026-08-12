import streamlit as st
from src.ingestion import load_and_chunk_pdfs
from src.indexing import build_vector_store
from src.generation import get_llm
from src.pipeline import build_pipeline, answer_question

st.set_page_config(page_title="Scheme Eligibility Assistant", page_icon="🏛️")

st.title(" Government Scheme Eligibility Assistant")
st.caption(
    "Ask about eligibility for PM-KISAN, Ayushman Bharat (PM-JAY), PMAY-Urban 2.0, "
    "or PM-KMY. Answers are grounded in official scheme guideline documents."
)


@st.cache_resource(show_spinner="Loading scheme documents and models (first run only)...")
def cached_pipeline():
    return build_pipeline()

vectorstore, chunks, llm = cached_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("e.g. I'm a farmer with 2 acres, am I eligible for any scheme?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching scheme documents..."):
            answer, docs, scheme_filter = answer_question(question, vectorstore, chunks, llm)
            st.markdown(answer)

            with st.expander(f"Sources used ({len(docs)} chunks)"):
                for doc in docs:
                    st.markdown(
                        f"- **{doc.metadata.get('scheme_name')}** "
                        f"(page {doc.metadata.get('page')})"
                    )

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.sidebar.markdown("###  Disclaimer")
st.sidebar.markdown(
    "This is a portfolio demonstration grounded in scheme guidelines as downloaded. "
    "Government schemes are periodically amended — verify eligibility on official portals "
    "before relying on this for real decisions."
)