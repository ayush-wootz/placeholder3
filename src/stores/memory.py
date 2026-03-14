"""Conversation memory store — maintains context across turns."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Turn:
    query: str
    answer: str
    sources: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MemoryStore:
    """In-memory conversation history, keyed by chat/user ID."""

    def __init__(self, max_turns: int = 50):
        self._max_turns = max_turns
        self._chats: dict[str, list[Turn]] = {}

    def add_turn(
        self, chat_id: str, query: str, answer: str, sources: list[str] | None = None
    ) -> None:
        if chat_id not in self._chats:
            self._chats[chat_id] = []
        self._chats[chat_id].append(Turn(query=query, answer=answer, sources=sources or []))
        # Evict oldest turns if over limit
        if len(self._chats[chat_id]) > self._max_turns:
            self._chats[chat_id] = self._chats[chat_id][-self._max_turns :]

    def get_history(self, chat_id: str, last_n: int | None = None) -> list[Turn]:
        turns = self._chats.get(chat_id, [])
        if last_n is not None:
            return turns[-last_n:]
        return list(turns)

    def get_context_summary(self, chat_id: str, last_n: int = 5) -> str:
        """Return a compact text summary of recent turns for LLM context."""
        turns = self.get_history(chat_id, last_n)
        if not turns:
            return ""
        lines = []
        for t in turns:
            lines.append(f"Q: {t.query}")
            lines.append(f"A: {t.answer[:200]}")
        return "\n".join(lines)

    def clear(self, chat_id: str) -> None:
        self._chats.pop(chat_id, None)
