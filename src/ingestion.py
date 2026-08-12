import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

RAW_PDF_DIR = Path("data/raw_pdfs")

# Map filenames to a clean scheme name 
SCHEME_NAME_MAP = {
    "OPERATIONAL GUIDELINES.pdf": "PM-KISAN",
    "Operation-Manual-for-AB-PM-JAY-April-2022.pdf": "Ayushman Bharat (PM-JAY)",
    "Operational-Guidelines-of-PMAY-U-2.pdf": "PMAY-Urban 2.0",
    "PM-KMY - Operational Guidelines.pdf": "PM Kisan Maandhan Yojana (PM-KMY)",
}


def load_and_chunk_pdfs(raw_pdf_dir: Path = RAW_PDF_DIR):
    """
    Loads every PDF in raw_pdf_dir, splits into chunks, and attaches
    metadata (scheme_name, source_file, page) to each chunk.
    Returns a list of LangChain Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,      # ~500-700 tokens (4 chars ≈ 1 token in English)
        chunk_overlap=300,    # prevents cutting eligibility clauses mid-sentence
        separators=["\n\n", "\n", ". ", " ", ""],  # tries paragraph breaks first, falls back
    )

    all_chunks = []

    pdf_files = list(raw_pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {raw_pdf_dir}. Add scheme PDFs there first.")

    for pdf_path in pdf_files:
        print(f"Loading {pdf_path.name}...")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()  # one Document per page, with page metadata already attached

        chunks = splitter.split_documents(pages)

        scheme_name = SCHEME_NAME_MAP.get(pdf_path.name, pdf_path.stem)

        for chunk in chunks:
            chunk.metadata["scheme_name"] = scheme_name
            chunk.metadata["source_file"] = pdf_path.name
            # chunk.metadata["page"] already set by PyPDFLoader

        all_chunks.extend(chunks)
        print(f"  → {len(chunks)} chunks from {pdf_path.name}")

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks
