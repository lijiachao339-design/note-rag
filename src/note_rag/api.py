"""FastAPI surface.

Kept intentionally thin: the retrieval logic lives in :mod:`note_rag.retriever` so that the
HTTP layer, the MCP server and the eval CLI all exercise the *same* code path. If your
eval used a different code path than production, your numbers are fiction.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from note_rag.config import Settings, get_settings
from note_rag.retriever import Retriever


class Hit(BaseModel):
    chunk_id: str
    note_path: str
    title: str
    heading_path: list[str]
    text: str
    score: float = Field(description="RRF fusion score")


class SearchResponse(BaseModel):
    query: str
    mode: str
    took_ms: float
    hits: list[Hit]


class MetricsRegistry:
    """Minimal in-process counters. Swap for Prometheus once you need dashboards."""

    def __init__(self) -> None:
        self.counters: dict[str, float] = {
            "search_requests_total": 0.0,
            "search_errors_total": 0.0,
            "search_latency_ms_sum": 0.0,
        }

    def observe(self, *, latency_ms: float, error: bool = False) -> None:
        self.counters["search_requests_total"] += 1
        self.counters["search_latency_ms_sum"] += latency_ms
        if error:
            self.counters["search_errors_total"] += 1

    def snapshot(self) -> dict[str, Any]:
        data = dict(self.counters)
        total = data["search_requests_total"] or 1.0
        data["search_latency_ms_avg"] = data["search_latency_ms_sum"] / total
        return data


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.retriever = Retriever.from_settings(settings)
        self.metrics = MetricsRegistry()


def _state(app: FastAPI) -> AppState:
    state: AppState = app.state.note_rag
    return state


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = AppState(settings)
        # Build the keyword index during startup so that /healthz and /metrics reflect real
        # readiness instead of looking healthy until the first search silently misbehaves.
        # (Retriever.search also lazy-builds as a safety net.)
        state.retriever.build_keyword_index()
        app.state.note_rag = state
        yield

    app = FastAPI(
        title="note-rag",
        version="0.1.0",
        description="Hybrid retrieval over an Obsidian vault (vector + BM25 + RRF).",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        state = _state(app)
        return {"status": "ok", "corpus": state.retriever.corpus_stats()}

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        state = _state(app)
        return state.metrics.snapshot() | state.retriever.index_stats()

    @app.get("/search", response_model=SearchResponse)
    def search(
        q: str = Query(min_length=1, max_length=512),
        k: int = Query(default=10, ge=1, le=100),
        mode: str = Query(default="hybrid", pattern="^(hybrid|vector|keyword|hybrid_rerank)$"),
    ) -> SearchResponse:
        state = _state(app)
        started = time.perf_counter()
        try:
            hits = state.retriever.search(q, k=k, mode=mode)  # type: ignore[arg-type]
        except Exception as exc:
            state.metrics.observe(latency_ms=(time.perf_counter() - started) * 1000, error=True)
            raise HTTPException(
                status_code=503, detail=f"retrieval backend unavailable: {exc}"
            ) from exc
        took_ms = (time.perf_counter() - started) * 1000
        state.metrics.observe(latency_ms=took_ms)
        return SearchResponse(
            query=q,
            mode=mode,
            took_ms=round(took_ms, 2),
            hits=[
                Hit(
                    chunk_id=hit.chunk_id,
                    note_path=hit.note_path,
                    title=hit.title,
                    heading_path=list(hit.heading_path),
                    text=hit.text,
                    score=hit.score,
                )
                for hit in hits
            ],
        )

    return app


app = create_app()
