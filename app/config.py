from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = "openai"
    google_api_key: str = ""
    openai_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    openai_model: str = "gpt-4.1-mini"

    embedding_model: str = "text-embedding-3-small"

    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_path: str = "vectorstore_db"
    qdrant_collection: str = "rag_chunks"

    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieve_k: int = 5
    rrf_k: int = 60
    bm25_corpus_path: str = "vectorstore_db/bm25_corpus.json"

    source_pdfs_dir: str = "data/source_pdfs"


settings = Settings()
