from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.vectorstore import close_client
from app.eval.tracing import configure_tracing, flush_traces


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start and stop hooks for the API process.

    Startup: enable LangSmith tracing when LANGSMITH_API_KEY is set.
    The OpenAI client and Qdrant client are created on the first
    /ingest or /ask call.

    Shutdown: flush traces, then close the local Qdrant client.
    """
    configure_tracing()
    yield
    flush_traces()
    close_client()


app = FastAPI(
    title="RAG Pipeline API",
    description=(
        "Phase 8 HTTP layer around the LangGraph graph. "
        "POST /ingest loads PDFs into Qdrant and BM25. "
        "POST /ask runs retrieve → rerank → prune → quality check "
        "(rewrite/retry) → generate, with prompt cache and LangSmith traces. "
        "Eval lives in the CLI: python -m app.cli eval."
    ),
    version="0.8.0",
    lifespan=lifespan,
)

app.include_router(router)
