"""
Core Bot — routes questions between proprietary and web sources,
then synthesizes a natural answer via LLM.

Data priority matrix:
  Internal question           → 100% proprietary
  General question + internal → Synthesize both
  General question            → 100% web
"""

from __future__ import annotations

import logging
import re

import httpx

from src.config import Config
from src.stores.memory import MemoryStore
from src.tools.search_proprietary import ProprietaryBackend, ProprietaryResult
from src.tools.search_web import WebSearchTool, WebResult
from src.utils.formatter import (
    enforce_personality,
    truncate,
    format_proprietary_results,
    format_web_results,
)

logger = logging.getLogger(__name__)


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
        config: Config | None = None,
    ):
        self.proprietary = proprietary_db
        self.web = web_search
        self.memory = memory or MemoryStore()
        self.config = config or Config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def answer(self, user_query: str, chat_id: str = "default") -> str:
        """Main entry-point: classify → route → LLM synthesize → format → remember."""

        is_internal = self.classify_as_internal(user_query)

        if is_internal:
            data_block, prop_results, web_results = await self._handle_internal(user_query)
        else:
            data_block, prop_results, web_results = await self._handle_general(user_query)

        # Build context from conversation memory
        context = self.memory.get_context_summary(chat_id, last_n=5)

        # Try LLM synthesis; fall back to formatted results if no API key
        if self.config.openai_api_key:
            response = await self._llm_synthesize(user_query, data_block, context)
        else:
            response = data_block or "No relevant data found."

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

    async def _handle_internal(self, query: str) -> tuple[str, list[ProprietaryResult], list[WebResult]]:
        """Workflow 1: internal question → proprietary first, web fallback."""
        prop_results = await self.proprietary.search(query, top_k=5)

        if prop_results:
            block = self._build_data_block(proprietary=prop_results)
            return block, prop_results, []

        # Fallback to web
        web_results = await self._web_search(query)
        block = self._build_data_block(web=web_results, note="No internal data found for this query.")
        return block, [], web_results

    async def _handle_general(self, query: str) -> tuple[str, list[ProprietaryResult], list[WebResult]]:
        """Workflow 2/3: general question → web + optional proprietary synthesis."""
        web_results = await self._web_search(query)
        prop_results = await self.proprietary.search(query, top_k=3)

        if prop_results and self._is_relevant(prop_results):
            block = self._build_data_block(web=web_results, proprietary=prop_results)
            return block, prop_results, web_results

        block = self._build_data_block(web=web_results)
        return block, [], web_results

    # ------------------------------------------------------------------
    # LLM synthesis
    # ------------------------------------------------------------------

    async def _llm_synthesize(self, query: str, data_block: str, context: str) -> str:
        """Call OpenAI to synthesize a natural answer from retrieved data."""
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            bot_name=self.config.bot_name,
            banned_words=", ".join(self.config.banned_words),
            banned_openings=", ".join(self.config.banned_openings),
            banned_closings=", ".join(self.config.banned_closings),
            max_length=self.config.max_message_length,
            context=context or "(No prior conversation)",
            data_block=data_block or "(No data retrieved)",
        )

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": query},
                        ],
                        "max_tokens": 400,
                        "temperature": 0.4,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            logger.exception("LLM call failed, falling back to formatted results")
            return data_block or "No relevant data found."

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

    def _build_data_block(
        self,
        proprietary: list[ProprietaryResult] | None = None,
        web: list[WebResult] | None = None,
        note: str | None = None,
    ) -> str:
        """Assemble the data context that gets fed to the LLM (or returned directly)."""
        parts: list[str] = []

        if proprietary:
            parts.append("*Internal data:*")
            parts.append(format_proprietary_results(proprietary))

        if web:
            if proprietary:
                parts.append("")
            parts.append("*Web results:*")
            parts.append(format_web_results(web))

        if note:
            parts.append(f"\n_{note}_")

        return "\n".join(parts) if parts else "No relevant data found."

    @staticmethod
    def _extract_sources(answer: str) -> list[str]:
        """Pull source references out of the formatted answer."""
        return re.findall(r"source:\s*([^)_]+)", answer)
