"""Runtime configuration.

Every setting is read from the environment with the ``NOTE_RAG_`` prefix, so the same
code runs locally (``.env``), in Docker (compose ``environment:``) and in CI.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NOTE_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    vault_path: Path = Field(default=Path("./vault"), description="Obsidian vault root")
    database_url: str = "postgresql://note_rag:note_rag@localhost:5432/note_rag"
    redis_url: str = "redis://localhost:6379/0"

    # Empty base_url => built-in offline hashing embedding (no API key needed)
    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_model: str = ""
    embed_dim: int = 384
    embed_batch_size: int = 64
    embed_timeout_s: float = 30.0

    # Retrieval defaults
    top_k: int = 10
    candidate_k: int = 50
    rrf_k: int = 60
    vector_weight: float = 1.0
    keyword_weight: float = 1.0

    # Chunking defaults
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 150

    @property
    def has_remote_embedder(self) -> bool:
        return bool(self.embed_base_url and self.embed_model)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
