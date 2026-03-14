"""Tool 1: search_proprietary — queries the proprietary vector DB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ProprietaryResult:
    content: str
    source: str
    relevance_score: float


class ProprietaryBackend(Protocol):
    """Any backend that can search proprietary data."""

    async def search(self, query: str, top_k: int = 5) -> list[ProprietaryResult]:
        ...


# ---------------------------------------------------------------------------
# Option A — Pinecone (vector DB)
# ---------------------------------------------------------------------------
class PineconeBackend:
    def __init__(self, api_key: str, index_name: str, environment: str, embed_fn):
        from pinecone import Pinecone

        pc = Pinecone(api_key=api_key)
        self._index = pc.Index(index_name)
        self._embed = embed_fn  # async (text) -> list[float]

    async def search(self, query: str, top_k: int = 5) -> list[ProprietaryResult]:
        vector = await self._embed(query)
        resp = self._index.query(vector=vector, top_k=top_k, include_metadata=True)
        return [
            ProprietaryResult(
                content=m["metadata"].get("content", ""),
                source=m["metadata"].get("source", "unknown"),
                relevance_score=m["score"],
            )
            for m in resp["matches"]
        ]


# ---------------------------------------------------------------------------
# Option B — PostgreSQL (structured DB, full-text search)
# ---------------------------------------------------------------------------
class PostgresBackend:
    def __init__(self, connection_url: str):
        self._url = connection_url

    async def search(self, query: str, top_k: int = 5) -> list[ProprietaryResult]:
        import asyncpg

        conn = await asyncpg.connect(self._url)
        try:
            rows = await conn.fetch(
                """
                SELECT content, source, ts_rank(tsv, plainto_tsquery($1)) AS score
                FROM documents
                WHERE tsv @@ plainto_tsquery($1)
                ORDER BY score DESC
                LIMIT $2
                """,
                query,
                top_k,
            )
            return [
                ProprietaryResult(
                    content=r["content"],
                    source=r["source"],
                    relevance_score=float(r["score"]),
                )
                for r in rows
            ]
        finally:
            await conn.close()


# ---------------------------------------------------------------------------
# Option C — Elasticsearch (document store)
# ---------------------------------------------------------------------------
class ElasticsearchBackend:
    def __init__(self, url: str, index: str = "documents"):
        self._url = url
        self._index = index

    async def search(self, query: str, top_k: int = 5) -> list[ProprietaryResult]:
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._url}/{self._index}/_search",
                json={
                    "size": top_k,
                    "query": {"multi_match": {"query": query, "fields": ["content", "title", "tags"]}},
                },
            )
            resp.raise_for_status()
            hits = resp.json()["hits"]["hits"]
            return [
                ProprietaryResult(
                    content=h["_source"].get("content", ""),
                    source=h["_source"].get("source", h["_id"]),
                    relevance_score=h["_score"],
                )
                for h in hits
            ]


# ---------------------------------------------------------------------------
# In-memory backend (for testing / bootstrapping)
# ---------------------------------------------------------------------------
class InMemoryBackend:
    """Simple keyword-match backend useful for local dev and tests."""

    def __init__(self, documents: list[dict] | None = None):
        self._docs: list[dict] = documents or []

    def add(self, content: str, source: str) -> None:
        self._docs.append({"content": content, "source": source})

    async def search(self, query: str, top_k: int = 5) -> list[ProprietaryResult]:
        query_lower = query.lower()
        scored = []
        for doc in self._docs:
            words = query_lower.split()
            matches = sum(1 for w in words if w in doc["content"].lower())
            if matches > 0:
                score = matches / len(words)
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            ProprietaryResult(content=d["content"], source=d["source"], relevance_score=s)
            for s, d in scored[:top_k]
        ]
