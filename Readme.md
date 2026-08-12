# Government Scheme Eligibility Assistant (RAG)

A Retrieval-Augmented Generation system that answers questions about eligibility for Indian government welfare schemes, grounded in official scheme guideline documents, with citations, so answers don't rely on the LLM's own (potentially wrong) knowledge of eligibility rules.


## Schemes covered

- **PM-KISAN** — income support for small/marginal farmers
- **Ayushman Bharat (PM-JAY)** — health insurance coverage
- **PMAY-Urban 2.0** — urban housing support
- **PM Kisan Maandhan Yojana (PM-KMY)** — farmer pension scheme

Source: official `.gov.in` scheme operational guideline PDFs.

## Architecture

```
User question
   ↓
Query Construction   — if the question names a specific scheme, filter retrieval to it
   ↓
Query Translation    — multi-query: LLM generates 3 reworded variants to widen recall
   ↓
Hybrid Retrieval     — dense (Chroma + bge-small-en embeddings) + sparse (BM25),
                        combined via weighted Reciprocal Rank Fusion
   ↓
Re-ranking           — cross-encoder (ms-marco-MiniLM-L-6-v2) scores and narrows
                        candidates down to the most relevant chunks
   ↓
Generation           — Groq (Llama 3.3 70B), strict prompt: answer only from
                        context, cite scheme name per claim, refuse if insufficient
   ↓
Answer + cited source chunks
```

## Tech stack

| Layer | Choice |
|---|---|
| LLM | Groq API (Llama 3.3 70B), free tier |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, free) |
| Vector DB | ChromaDB (local, persisted) |
| Sparse retrieval | BM25 (`rank_bm25`) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Framework | LangChain (LCEL) |
| UI | Streamlit |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision) |

## Project structure

```
src/
├── ingestion.py          # PDF loading, chunking, metadata attachment
├── indexing.py           # Embedding + Chroma vector store, BM25 index
├── retrieval.py          # Hybrid retriever (dense+sparse) + reranking
├── query_translation.py  # Multi-query generation
├── query_construction.py # Scheme-name filter extraction
├── generation.py         # Grounded answer generation via Groq
└── pipeline.py           # Orchestrates the full flow

app.py                    # Streamlit chat interface
eval/
├── test_questions.json   # Fixed eval question set
└── run_eval.py           # RAGAS scoring
data/raw_pdfs/             # Source scheme PDFs (gitignored)
```

## Setup

```bash
pip install -r requirements.txt --break-system-packages

# Add scheme PDFs to data/raw_pdfs/
# Copy .env.example to .env and add your GROQ_API_KEY

streamlit run app.py
```

## Design decisions

- **Chunking**: recursive character splitting, ~2000 chars (~500-700 tokens) with 300-char overlap, to avoid splitting eligibility clauses mid-sentence.
- **Multi-query over RAG Fusion**: user situations are phrased inconsistently with how scheme documents describe eligibility (e.g. "2 acres" vs "cultivable land up to 2 hectare")  multi-query directly targets this vocabulary-mismatch recall problem.
- **Hybrid search**: BM25 is included alongside dense embeddings specifically because scheme documents contain exact terms  amounts, dates, scheme names that embeddings can blur but keyword search catches precisely.
- **Query construction kept lightweight**: plain string/alias matching rather than an LLM function-calling approach, since the corpus only spans 4 known schemes sufficient for this scope, though it doesn't handle multi-scheme comparison questions (see Limitations).
- **`temperature=0`** for generation (deterministic, grounded answers) vs. default temperature for multi-query generation (some variation in phrasing is desirable there).

## Known limitations

- **Retrieval/generation speed**: the current pipeline noticeably slow per question each question involves multiple LLM calls (multi-query generation + final answer) plus retrieval across 4 query variants. Startup (PDF parsing, embedding, model loading) is not yet fully cached outside the Streamlit app; the underlying scripts still redo setup work on each run. This is a known, unresolved performance gap not yet optimized.
- **Query construction** only extracts a single scheme per question a question comparing two schemes will only filter to one, potentially missing the other's context.
- **Data freshness**: this system is grounded in scheme guideline PDFs as downloaded at build time. Government schemes are periodically amended; this is a portfolio demonstration, not a live/authoritative eligibility source. Always verify against official scheme portals.
- **Streamlit caching**: the app uses `st.cache_resource` to load the pipeline once per session, but the interaction model (full script rerun per user action) is still being worked through — noted here as an area for further learning/refinement.

## Future work

- Complete RAGAS evaluation run and report before/after metrics (naive dense-only vs. hybrid+rerank)
- Tune hybrid retriever weights (currently 0.5/0.5 dense/sparse) based on eval results
- Multi-scheme query construction
- Caching/performance pass on the core pipeline scripts outside the Streamlit app