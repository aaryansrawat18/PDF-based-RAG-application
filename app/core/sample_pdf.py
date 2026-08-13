from pathlib import Path

import pymupdf

_PAGES = [
    (
        "What is RAG?",
        (
            "Retrieval-Augmented Generation (RAG) is a technique that combines document "
            "retrieval with a large language model. Instead of answering only from model "
            "weights, the system first searches a knowledge base, then generates an answer "
            "grounded in the retrieved passages.\n\n"
            "A typical RAG system stores text chunks in a vector database. At query time it "
            "embeds the user question, finds similar chunks, and passes those chunks to the "
            "LLM as context. This reduces hallucinations on private or recent documents "
            "that the model was never trained on.\n\n"
            "Basic metadata such as page number and document name should travel with each "
            "chunk so answers can cite sources."
        ),
    ),
    (
        "The RAG pipeline",
        (
            "The core RAG pipeline has five steps. First, load PDFs and extract page text. "
            "Second, split pages into overlapping chunks so related sentences stay together. "
            "Third, embed each chunk with a sentence transformer such as BGE or E5. "
            "Fourth, upsert vectors into Qdrant. Fifth, at query time retrieve the top-k "
            "chunks and generate an answer.\n\n"
            "Chunk size and overlap matter. Chunks around 800 characters with 150 characters "
            "of overlap are a practical starting point for technical PDFs. Too-small chunks "
            "lose context. Too-large chunks dilute similarity search.\n\n"
            "LangGraph can wire retrieve and generate as two linear nodes in phase 1. Later "
            "phases add hybrid search, reranking, pruning, and retry edges."
        ),
    ),
    (
        "Advanced RAG techniques",
        (
            "Advanced RAG improves retrieval quality before the LLM sees context. Hybrid "
            "search fuses dense vector results with BM25 keyword matches using Reciprocal "
            "Rank Fusion (RRF). A cross-encoder reranker then scores query-chunk pairs more "
            "accurately than bi-encoder similarity alone.\n\n"
            "Context pruning drops low-score, duplicate, or token-heavy passages so the "
            "prompt stays small. Metadata filters restrict search to a section, document, "
            "or page range. Query rewrite retries retrieval when the first context is weak.\n\n"
            "Prompt caching keeps a stable instruction prefix so the provider can reuse "
            "the KV cache. Observability with LangSmith records node spans for retrieve, "
            "rerank, prune, and generate."
        ),
    ),
]


def ensure_sample_pdf(path: str | Path) -> Path:
    """Write a small 3-page RAG explainer PDF if it does not already exist."""
    pdf_path = Path(path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return pdf_path

    doc = pymupdf.open()
    for title, body in _PAGES:
        page = doc.new_page()
        page.insert_text((72, 72), title, fontsize=18)
        page.insert_textbox(
            pymupdf.Rect(72, 110, 540, 760),
            body,
            fontsize=11,
            align=0,
        )
    doc.save(pdf_path)
    doc.close()
    return pdf_path
