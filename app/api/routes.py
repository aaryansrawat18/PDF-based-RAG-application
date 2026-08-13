import time

from fastapi import APIRouter, HTTPException

from app.core.ingest import ingest_pdfs
from app.core.rag_chain import run_rag
from app.models.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestDocumentResult,
    IngestRequest,
    IngestResponse,
    Source,
)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
def health() -> HealthResponse:
    """Simple liveness check. Does not talk to Qdrant or the LLM."""
    return HealthResponse(status="ok")


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Load PDFs into Qdrant",
)
def ingest(request: IngestRequest | None = None) -> IngestResponse:
    """Step 1 of using the API: put PDF chunks into the vector store.

    This route does not talk to Qdrant or BM25 itself. It only calls
    ingest_pdfs(), which already does: load PDF → chunk → embed →
    upsert Qdrant → rebuild the BM25 corpus for that document.

    Body can be omitted, {}, or {"pdf_path": "..."}.
    """
    if request is None:
        request = IngestRequest()

    try:
        results = ingest_pdfs(request.pdf_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ingest failed: {exc}",
        ) from exc

    documents = [
        IngestDocumentResult(
            document=item["document"],
            pages=item["pages"],
            chunks=item["chunks"],
            tables=item.get("tables", 0),
            figures=item.get("figures", 0),
        )
        for item in results
    ]

    if not documents:
        message = "No PDFs were ingested."
    elif len(documents) == 1:
        message = f"Ingested {documents[0].document}."
    else:
        message = f"Ingested {len(documents)} PDFs."

    return IngestResponse(message=message, documents=documents)


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question",
)
def ask(request: AskRequest) -> AskResponse:
    """Step 2 of using the API: retrieve, rerank, prune, then generate.

    This route does not call LangGraph nodes directly. It only calls
    run_rag(question, filters), which is a thin wrapper around graph.invoke(...).
    Graph: hybrid retrieve → rerank → prune → quality check
    (rewrite + retrieve again if context is weak) → generate
    (one retry if the answer is a refusal). LangSmith records node
    spans when LANGSMITH_API_KEY is set.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="question must be a non-empty string",
        )

    filters = None
    if request.filters is not None:
        filters = request.filters.model_dump(exclude_none=True) or None

    started = time.perf_counter()
    try:
        result = run_rag(question, filters=filters)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ask failed: {exc}",
        ) from exc

    raw_sources = result.get("sources") or []
    sources = [
        Source(
            document=item.get("document"),
            page=item.get("page"),
            section=item.get("section"),
            chunk_id=item.get("chunk_id"),
            score=item.get("score"),
            content_type=item.get("content_type"),
        )
        for item in raw_sources
    ]

    latency_ms = result.get("latency_ms")
    if latency_ms is None:
        latency_ms = int((time.perf_counter() - started) * 1000)

    return AskResponse(
        answer=result.get("answer", ""),
        sources=sources,
        latency_ms=int(latency_ms),
    )
