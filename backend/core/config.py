from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDER_MARKERS = ("replace-me", "your-resource", "example")


def configured(*values: str | None) -> bool:
    return (
        bool(values)
        and all(values)
        and not any(
            marker in str(value).lower() for value in values for marker in PLACEHOLDER_MARKERS
        )
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    app_name: str = "Chart Assistant"
    log_level: str = "INFO"
    prompt_dir: Path = ROOT / "backend" / "prompts"
    artifact_dir: Path = ROOT / ".artifacts" / "ingestion"

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str | None = None
    azure_openai_embedding_deployment: str | None = None

    azure_search_endpoint: str | None = None
    azure_search_admin_key: str | None = None
    azure_search_query_key: str | None = None
    azure_search_index_name: str = "healthchat-documents-v1"

    top_k: int = Field(4, alias="RAG_TOP_K", ge=1, le=10)
    candidate_limit: int = Field(12, alias="RAG_CANDIDATE_LIMIT", ge=4, le=50)
    context_token_budget: int = Field(900, alias="RAG_CONTEXT_TOKEN_BUDGET", ge=200)
    semantic_reranker_threshold: float = Field(
        1.5, alias="RAG_SEMANTIC_RERANKER_THRESHOLD", ge=0, le=4
    )
    bm25_min_score: float = Field(0.75, alias="RAG_BM25_MIN_SCORE", ge=0)
    vector_min_similarity: float = Field(0.55, alias="RAG_VECTOR_MIN_SIMILARITY", ge=-1, le=1)
    target_chunk_tokens: int = 180
    max_chunk_tokens: int = 240
    chunk_overlap_tokens: int = 30
    max_upload_files: int = 25
    max_file_bytes: int = 5 * 1024 * 1024
    max_total_upload_bytes: int = 25 * 1024 * 1024

    @property
    def chat_ready(self) -> bool:
        return configured(
            self.azure_openai_endpoint,
            self.azure_openai_api_key,
            self.azure_openai_chat_deployment,
        )

    @property
    def embeddings_ready(self) -> bool:
        return configured(
            self.azure_openai_endpoint,
            self.azure_openai_api_key,
            self.azure_openai_embedding_deployment,
        )

    @property
    def search_query_ready(self) -> bool:
        return configured(
            self.azure_search_endpoint,
            self.azure_search_query_key,
            self.azure_search_index_name,
        )

    @property
    def search_admin_ready(self) -> bool:
        return configured(
            self.azure_search_endpoint,
            self.azure_search_admin_key,
            self.azure_search_index_name,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
