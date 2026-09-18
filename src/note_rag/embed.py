"""Embedding clients.

Two implementations:

* :class:`HashingEmbedder` -- the "hashing trick": token ids are hashed into a fixed
  number of buckets with a signed accumulator. It needs no API key, is deterministic, and
  is good enough to build and debug the whole pipeline (and to run CI). It is *not* a
  semantic model: it will fail on paraphrase queries, which is exactly the failure mode
  your eval set should expose before you spend money on a real embedder.
* :class:`RemoteEmbedder` -- any OpenAI-compatible ``/v1/embeddings`` endpoint.

Both honour a local cache keyed by ``content_hash`` so re-ingesting an unchanged vault
costs zero embedding calls. That cache is what makes "incremental indexing" real.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from typing import Protocol

import httpx

from note_rag.bm25 import tokenize


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vec))
    if norm == 0.0:
        return vec
    return [value / norm for value in vec]


class HashingEmbedder:
    """Deterministic offline embedder (unigram/bigram feature hashing + L2 norm)."""

    def __init__(self, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vec[index] += sign
        return _normalise(vec)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]


class RemoteEmbedder:
    """OpenAI-compatible embeddings client with batching."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dim: int,
        batch_size: int = 64,
        timeout_s: float = 30.0,
    ) -> None:
        if not base_url or not model:
            raise ValueError("base_url and model are required for RemoteEmbedder")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.timeout_s = timeout_s

    def _embed_batch(self, client: httpx.Client, batch: Sequence[str]) -> list[list[float]]:
        response = client.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": list(batch)},
        )
        response.raise_for_status()
        payload = response.json()
        rows = sorted(payload["data"], key=lambda row: row["index"])
        vectors = [_normalise([float(v) for v in row["embedding"]]) for row in rows]
        if vectors and len(vectors[0]) != self.dim:
            raise ValueError(
                f"embedding dim mismatch: model returned {len(vectors[0])}, "
                f"NOTE_RAG_EMBED_DIM is {self.dim}"
            )
        return vectors

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with httpx.Client(timeout=self.timeout_s, headers=headers) as client:
            for start in range(0, len(texts), self.batch_size):
                out.extend(self._embed_batch(client, texts[start : start + self.batch_size]))
        return out


class CachedEmbedder:
    """Wraps an embedder with a dict cache. Swap in Redis once chunks exceed memory."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._cache: dict[str, list[float]] = {}
        self.dim = inner.dim
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key_for(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [self.key_for(text) for text in texts]
        pending = [
            (key, text) for key, text in zip(keys, texts, strict=True) if key not in self._cache
        ]
        if pending:
            self.misses += len(pending)
            fresh = self._inner.embed([text for _, text in pending])
            for (key, _), vec in zip(pending, fresh, strict=True):
                self._cache[key] = vec
        else:
            self.hits += len(keys)
        return [self._cache[key] for key in keys]

    def stats(self) -> dict[str, int]:
        return {
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_size": len(self._cache),
        }


def build_embedder(
    *,
    base_url: str = "",
    api_key: str = "",
    model: str = "",
    dim: int = 384,
    batch_size: int = 64,
) -> Embedder:
    """Factory used by the CLI and the API.

    Falls back to :class:`HashingEmbedder` when no remote endpoint is configured, so a
    fresh clone runs end to end with no credentials.
    """
    if base_url and model:
        return RemoteEmbedder(
            base_url=base_url, api_key=api_key, model=model, dim=dim, batch_size=batch_size
        )
    return HashingEmbedder(dim=dim)


def embed_all(embedder: Embedder, texts: Iterable[str]) -> list[list[float]]:
    """Convenience helper for one-shot use."""
    return embedder.embed(list(texts))
