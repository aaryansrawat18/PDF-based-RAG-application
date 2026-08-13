from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Body for POST /ingest.

    pdf_path is optional:
    - If you send a path, we ingest that one PDF.
    - If you omit it (or send null), we ingest every PDF in data/source_pdfs/.
    """

    pdf_path: str | None = Field(
        default=None,
        description="Path to one PDF, for example data/source_pdfs/Document.pdf",
        examples=["data/source_pdfs/Document.pdf"],
    )


class IngestDocumentResult(BaseModel):
    """What happened for one PDF during ingest."""

    document: str
    pages: int
    chunks: int
    tables: int = 0
    figures: int = 0


class IngestResponse(BaseModel):
    """Body returned by POST /ingest."""

    message: str
    documents: list[IngestDocumentResult]


class AskRequest(BaseModel):
    """Body for POST /ask."""

    question: str = Field(
        ...,
        min_length=1,
        description="The user question to send through the RAG graph.",
        examples=["What is RAG?"],
    )


class Source(BaseModel):
    """One retrieved chunk that the answer is based on."""

    document: str | None = None
    page: int | None = None
    section: str | None = None
    chunk_id: str | None = None
    score: float | None = None
    content_type: str | None = None


class AskResponse(BaseModel):
    """Body returned by POST /ask."""

    answer: str
    sources: list[Source]
    latency_ms: int = Field(
        ...,
        ge=0,
        description="Graph invoke wall time in milliseconds.",
        examples=[1420],
    )


class HealthResponse(BaseModel):
    """Body returned by GET /health."""

    status: str
