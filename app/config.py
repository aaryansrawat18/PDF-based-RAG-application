from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"

    embedding_model: str = "text-embedding-3-small"

    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_path: str = "vectorstore_db"
    qdrant_collection: str = "rag_chunks"

    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieve_k: int = 20
    rerank_k: int = 10
    prune_k: int = 5
    rrf_k: int = 60
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    prune_score_threshold: float = 0.0
    prune_overlap_threshold: float = 0.8
    prune_max_tokens: int = 2000
    bm25_corpus_path: str = "vectorstore_db/bm25_corpus.json"

    rewrite_model: str = "gpt-5.4-nano"
    max_retrieve_retries: int = 2
    max_generate_retries: int = 1
    context_min_chunks: int = 1
    context_score_threshold: float = 0.5

    prompt_cache_key: str = "rag-pipeline-v1"
    prompt_cache_retention: str = "in_memory"

    langsmith_api_key: str = ""
    langsmith_project: str = "rag-pipeline"
    langsmith_tracing: bool = True
    langsmith_endpoint: str = ""

    source_pdfs_dir: str = "data/source_pdfs"


settings = Settings()
