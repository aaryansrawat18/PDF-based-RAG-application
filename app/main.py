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
        "Phase 3 HTTP layer around the LangGraph graph. "
        "POST /ingest loads PDFs. POST /ask runs retrieve → generate "
        "with optional metadata filters."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

app.include_router(router)
