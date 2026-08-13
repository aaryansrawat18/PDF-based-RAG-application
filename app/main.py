from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.vectorstore import close_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start and stop hooks for the API process.

    Startup: nothing heavy. The OpenAI client and Qdrant client are
    created on the first /ingest or /ask call.

    Shutdown: close the local Qdrant client so the process can exit cleanly.
    """
    yield
    close_client()


app = FastAPI(
    title="RAG Pipeline API",
    description=(
        "Phase 4 HTTP layer around the LangGraph graph. "
        "POST /ingest loads PDFs into Qdrant and BM25. "
        "POST /ask runs hybrid retrieve (vector + BM25 + RRF) → generate."
    ),
    version="0.4.0",
    lifespan=lifespan,
)

app.include_router(router)
