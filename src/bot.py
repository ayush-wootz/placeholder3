"""
Core Bot — routes questions between proprietary and web sources.

Data priority matrix:
  Internal question           → 100% proprietary
  General question + internal → Synthesize both
  General question            → 100% web
"""

from __future__ import annotations

from src.config import Config
from src.stores.memory import MemoryStore
from src.tools.search_proprietary import ProprietaryBackend, ProprietaryResult
from src.tools.search_web import WebSearchTool, WebResult
from src.tools.web_fetch import web_fetch
from src.tools.schedule_task import TaskScheduler
from src.utils.formatter import (
    enforce_personality,
    truncate,
    format_proprietary_results,
    format_web_results,
)


SYSTEM_PROMPT_TEMPLATE = """\
You are {bot_name}. A research-first agent that pulls from proprietary data first, web second.

Rules:
- Lead with the answer. First sentence = the insight.
- Use numbers, not adjectives. Use names, not descriptions.
- Match the user's energy.
- Never use these words: {banned_words}
- Never start with: {banned_openings}
- Never end with: {banned_closings}
- Never mention your search process.
- Format for messaging apps: *bold* for key terms, > for stats, - for bullets (3-5 max), → for actions.
- Keep responses under {max_length} characters.

Previous conversation context:
{context}

{data_block}

Answer the user's question using ONLY the data above. Cite whether data is internal or external.
"""


class Bot:
    def __init__(
        self,
        proprietary_db: ProprietaryBackend,
        web_search: WebSearchTool | None = None,
        memory: MemoryStore | None = None,
        scheduler: TaskScheduler | None = None,
        config: Config | None = None,
    ):
        self.proprietary = proprietary_db
        self.web = web_search
        self.memory = memory or MemoryStore()
        self.scheduler = scheduler
        self.config = config or Config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def answer(self, user_query: str, chat_id: str = "default") -> str:
        """Main entry-point: classify → route → format → remember."""

        is_internal = self.classify_as_internal(user_query)

        if is_internal:
            response = await self._handle_internal(user_query)
        else:
            response = await self._handle_general(user_query)

        # Enforce personality & length
        response = enforce_personality(response, self.config)
        response = truncate(response, self.config.max_message_length)

        # Persist to memory
        sources = self._extract_sources(response)
        self.memory.add_turn(chat_id, user_query, response, sources)

        return response

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def classify_as_internal(self, query: str) -> bool:
        """Return True if the query is about internal/project data."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.config.internal_keywords)

    async def _handle_internal(self, query: str) -> str:
        """Workflow 1: internal question → proprietary first, web fallback."""
        prop_results = await self.proprietary.search(query, top_k=5)

        if prop_results:
            return self._build_answer(proprietary=prop_results)

        # Fallback to web
        web_results = await self._web_search(query)
        return self._build_answer(web=web_results, note="No internal data found for this query.")

    async def _handle_general(self, query: str) -> str:
        """Workflow 2/3: general question → web + optional proprietary synthesis."""
        web_results = await self._web_search(query)
        prop_results = await self.proprietary.search(query, top_k=3)

        if prop_results and self._is_relevant(prop_results):
            return self._build_answer(web=web_results, proprietary=prop_results)

        return self._build_answer(web=web_results)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _web_search(self, query: str) -> list[WebResult]:
        if self.web is None:
            return []
        try:
            return await self.web.search(query)
        except Exception:
            return []

    @staticmethod
    def _is_relevant(results: list[ProprietaryResult], threshold: float = 0.3) -> bool:
        return any(r.relevance_score >= threshold for r in results)

    def _build_answer(
        self,
        proprietary: list[ProprietaryResult] | None = None,
        web: list[WebResult] | None = None,
        note: str | None = None,
    ) -> str:
        """Assemble a response from available data sources."""
        parts: list[str] = []

        if proprietary:
            parts.append(format_proprietary_results(proprietary))

        if web:
            if proprietary:
                parts.append("")  # blank line separator
            parts.append(format_web_results(web))

        if note:
            parts.append(f"\n_{note}_")

        return "\n".join(parts) if parts else "No relevant data found."

    @staticmethod
    def _extract_sources(answer: str) -> list[str]:
        """Pull source references out of the formatted answer."""
        import re
        return re.findall(r"source:\s*([^)_]+)", answer)
