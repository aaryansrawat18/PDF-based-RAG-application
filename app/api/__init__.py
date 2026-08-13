"""HTTP routes. Business logic stays in app.core / app.graph.

Flow: uvicorn app.main:app → include_router(routes.router)
  → /ingest → ingest_pdfs
  → /ask    → run_rag → LangGraph
"""
