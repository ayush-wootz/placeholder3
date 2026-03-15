"""Tool 1: search_proprietary — queries the ZAI Postgres + pgvector DB."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, Callable

import httpx

logger = logging.getLogger(__name__)


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
# Embedding helpers
# ---------------------------------------------------------------------------
async def openai_embed(text: str, api_key: str, model: str = "text-embedding-3-small") -> list[float]:
    """Get an embedding vector from OpenAI."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"input": text, "model": model},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


async def gemini_embed(
    text: str,
    api_key: str,
    model: str = "gemini-embedding-001",
    dims: int = 1536,
) -> list[float]:
    """Get an embedding vector from Gemini."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={api_key}"
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            url,
            json={
                "content": {"parts": [{"text": text}]},
                "taskType": "RETRIEVAL_QUERY",
                "outputDimensionality": dims,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]


def _vec_literal(vec: list[float]) -> str:
    """Convert a float list to pgvector literal format: '[0.1,0.2,...]'"""
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


# ---------------------------------------------------------------------------
# ZAI Postgres + pgvector backend
# ---------------------------------------------------------------------------
class ZaiPgvectorBackend:
    """
    Searches the ZAI database across all vector tables:
    - incident_vectors  (checkin incidents)
    - ccp_vectors       (CCP document chunks)
    - dashboard_vectors (dashboard updates)
    - glide_kb_vectors  (Glide knowledge base chunks)

    Uses cosine similarity via pgvector's <=> operator.
    """

    def __init__(
        self,
        connection_url: str,
        embed_fn: Callable,
        tenant_id: str = "",
    ):
        self._url = connection_url
        self._embed = embed_fn  # async (text) -> list[float]
        self._tenant_id = tenant_id

    async def search(self, query: str, top_k: int = 5) -> list[ProprietaryResult]:
        import asyncpg

        vector = await self._embed(query)
        vec_lit = _vec_literal(vector)

        conn = await asyncpg.connect(self._url)
        try:
            results: list[ProprietaryResult] = []

            # Search all 4 vector tables, collect top matches from each
            per_table = max(top_k, 3)

            # 1. Incident vectors
            rows = await conn.fetch(
                f"""
                SELECT summary_text AS content,
                       'incident/' || checkin_id AS source,
                       1 - (embedding <=> $1::vector) AS score
                FROM incident_vectors
                {"WHERE tenant_id = $3" if self._tenant_id else ""}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                vec_lit,
                per_table,
                *([self._tenant_id] if self._tenant_id else []),
            )
            for r in rows:
                results.append(ProprietaryResult(
                    content=r["content"],
                    source=r["source"],
                    relevance_score=float(r["score"]),
                ))

            # 2. CCP vectors (document chunks)
            rows = await conn.fetch(
                f"""
                SELECT chunk_text AS content,
                       COALESCE(source_ref, 'ccp/' || ccp_id) AS source,
                       1 - (embedding <=> $1::vector) AS score
                FROM ccp_vectors
                {"WHERE tenant_id = $3" if self._tenant_id else ""}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                vec_lit,
                per_table,
                *([self._tenant_id] if self._tenant_id else []),
            )
            for r in rows:
                results.append(ProprietaryResult(
                    content=r["content"],
                    source=r["source"],
                    relevance_score=float(r["score"]),
                ))

            # 3. Dashboard vectors
            rows = await conn.fetch(
                f"""
                SELECT update_message AS content,
                       'dashboard' AS source,
                       1 - (embedding <=> $1::vector) AS score
                FROM dashboard_vectors
                {"WHERE tenant_id = $3" if self._tenant_id else ""}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                vec_lit,
                per_table,
                *([self._tenant_id] if self._tenant_id else []),
            )
            for r in rows:
                results.append(ProprietaryResult(
                    content=r["content"],
                    source=r["source"],
                    relevance_score=float(r["score"]),
                ))

            # 4. Glide KB vectors (knowledge base)
            rows = await conn.fetch(
                f"""
                SELECT v.chunk_text AS content,
                       COALESCE(i.title, i.table_name || '/' || i.row_id) AS source,
                       1 - (v.embedding <=> $1::vector) AS score
                FROM glide_kb_vectors v
                JOIN glide_kb_items i
                  ON v.tenant_id = i.tenant_id AND v.item_id = i.item_id
                {"WHERE v.tenant_id = $3" if self._tenant_id else ""}
                ORDER BY v.embedding <=> $1::vector
                LIMIT $2
                """,
                vec_lit,
                per_table,
                *([self._tenant_id] if self._tenant_id else []),
            )
            for r in rows:
                results.append(ProprietaryResult(
                    content=r["content"],
                    source=r["source"],
                    relevance_score=float(r["score"]),
                ))

            # Sort all results by score, return top_k
            results.sort(key=lambda r: r.relevance_score, reverse=True)
            return results[:top_k]

        except Exception:
            logger.exception("pgvector search failed")
            return []
        finally:
            await conn.close()


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
