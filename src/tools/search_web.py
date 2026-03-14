"""Tool 2: search_web — web search with rate limiting."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
    source: str


class WebSearchTool:
    """Searches the web via Serper (Google SERP API). Rate-limited."""

    def __init__(self, api_key: str, rate_limit: int = 5):
        self._api_key = api_key
        self._rate_limit = rate_limit  # calls per minute
        self._call_times: list[float] = []

    def _enforce_rate_limit(self) -> None:
        now = time.time()
        self._call_times = [t for t in self._call_times if now - t < 60]
        if len(self._call_times) >= self._rate_limit:
            wait = 60 - (now - self._call_times[0])
            if wait > 0:
                time.sleep(wait)
        self._call_times.append(time.time())

    async def search(self, query: str, num_results: int = 5) -> list[WebResult]:
        self._enforce_rate_limit()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": self._api_key, "Content-Type": "application/json"},
                json={"q": query, "num": num_results},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

        results: list[WebResult] = []
        for item in data.get("organic", [])[:num_results]:
            results.append(
                WebResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source=item.get("link", ""),
                )
            )
        return results
