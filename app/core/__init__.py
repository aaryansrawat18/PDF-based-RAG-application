"""Core building blocks used by ingest and the LangGraph nodes.

Ingest order:  pdf_loader → chunking → embeddings → vectorstore + bm25
Ask retrieve:  embeddings + vectorstore + bm25 → hybrid → reranker → pruning
Ask generate:  prompts → llm
"""
