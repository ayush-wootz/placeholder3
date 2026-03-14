"""Tool 3: web_fetch — fetches and extracts text from a URL."""

from __future__ import annotations

import re

import httpx


async def web_fetch(url: str, timeout: int = 15) -> str | None:
    """Fetch *url* and return extracted text, or None on failure."""
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(url, timeout=timeout)
            resp.raise_for_status()
            html = resp.text
    except (httpx.HTTPError, httpx.TimeoutException):
        return None

    return _extract_text(html)


def _extract_text(html: str) -> str:
    """Naive HTML → plain-text extractor."""
    # Remove script/style blocks
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
