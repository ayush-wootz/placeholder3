"""Formatting & personality enforcement for bot responses."""

from __future__ import annotations

import re
from src.config import Config


def enforce_personality(text: str, config: Config) -> str:
    """Strip banned words/phrases and enforce style rules."""
    # Remove banned words (case-insensitive, whole-word)
    for word in config.banned_words:
        text = re.sub(rf"\b{re.escape(word)}\w*\b", "", text, flags=re.IGNORECASE)

    # Remove banned openings
    for opening in config.banned_openings:
        pattern = rf"^{re.escape(opening)}[!.,]?\s*"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove banned closings
    for closing in config.banned_closings:
        pattern = rf"\s*{re.escape(closing)}[!.]?\s*$"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Collapse extra whitespace
    text = re.sub(r"  +", " ", text).strip()
    return text


def truncate(text: str, max_length: int) -> str:
    """Truncate to max_length, breaking at last sentence boundary."""
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    # Try to break at last sentence end
    last_period = truncated.rfind(".")
    if last_period > max_length // 2:
        return truncated[: last_period + 1]
    return truncated.rstrip() + "..."


def format_proprietary_results(results: list) -> str:
    """Format proprietary search results into a citation-ready block."""
    if not results:
        return ""
    lines = []
    for r in results:
        lines.append(f"- {r.content} _(source: {r.source})_")
    return "\n".join(lines)


def format_web_results(results: list) -> str:
    """Format web search results."""
    if not results:
        return ""
    lines = []
    for r in results:
        lines.append(f"- {r.title}: {r.snippet}")
    return "\n".join(lines)
