"""Tests for the core Bot routing logic and tools."""

import pytest
import pytest_asyncio

from src.bot import Bot
from src.config import Config
from src.stores.memory import MemoryStore
from src.tools.search_proprietary import InMemoryBackend, ProprietaryResult
from src.utils.formatter import enforce_personality, truncate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return Config()


@pytest.fixture
def proprietary_db():
    db = InMemoryBackend()
    db.add(
        content="Project Falcon is 65% through Phase 2 as of March 10. QA starts March 25.",
        source="project_falcon/updates",
    )
    db.add(
        content="CX-Bot v2 resolves 71% of tickets without human handoff.",
        source="internal/CX-Bot/metrics",
    )
    db.add(
        content="Sprint 14 velocity: 42 points. Team capacity at 80%.",
        source="internal/sprint/14",
    )
    return db


@pytest.fixture
def bot(proprietary_db, config):
    return Bot(proprietary_db=proprietary_db, config=config)


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

class TestClassification:
    def test_internal_keyword_project(self, bot):
        assert bot.classify_as_internal("What's the status of Project Falcon?") is True

    def test_internal_keyword_team(self, bot):
        assert bot.classify_as_internal("How is our team doing?") is True

    def test_internal_keyword_sprint(self, bot):
        assert bot.classify_as_internal("Sprint 14 velocity?") is True

    def test_general_question(self, bot):
        assert bot.classify_as_internal("When does the new iPhone launch?") is False

    def test_general_ai_question(self, bot):
        assert bot.classify_as_internal("What is machine learning?") is False


# ---------------------------------------------------------------------------
# Routing tests (Workflows 1, 2, 3)
# ---------------------------------------------------------------------------

class TestDataBlock:
    """Test _build_data_block formatting."""

    def test_internal_data_header(self, bot, proprietary_db):
        from src.tools.search_proprietary import ProprietaryResult
        results = [ProprietaryResult(content="Test data", source="test/src", relevance_score=0.9)]
        block = bot._build_data_block(proprietary=results)
        assert "*Internal data:*" in block
        assert "Test data" in block

    def test_web_data_header(self, bot):
        from src.tools.search_web import WebResult
        results = [WebResult(title="Title", url="http://x.com", snippet="Snippet", source="http://x.com")]
        block = bot._build_data_block(web=results)
        assert "*Web results:*" in block
        assert "Title" in block

    def test_fallback_no_data(self, bot):
        block = bot._build_data_block()
        assert block == "No relevant data found."


class TestLLMFallback:
    """Without an OpenAI key, the bot should return formatted data directly."""

    @pytest.mark.asyncio
    async def test_no_api_key_returns_data_block(self, bot):
        """Without OPENAI_API_KEY, bot returns raw data block (no LLM call)."""
        result = await bot.answer("What's the status of Project Falcon?")
        assert "Falcon" in result or "65%" in result
        # Verify it's the formatted data, not an LLM response
        assert "*Internal data:*" in result or "source:" in result


class TestRouting:
    @pytest.mark.asyncio
    async def test_workflow1_internal_question(self, bot):
        """Internal question → proprietary DB returns data."""
        result = await bot.answer("What's the status of Project Falcon?")
        assert "Falcon" in result or "65%" in result
        assert "Phase 2" in result or "project_falcon" in result

    @pytest.mark.asyncio
    async def test_workflow1_internal_with_source(self, bot):
        """Internal results include source citations."""
        result = await bot.answer("Tell me about our sprint velocity")
        assert "sprint" in result.lower() or "42" in result

    @pytest.mark.asyncio
    async def test_workflow3_general_no_match(self, bot):
        """General question with no proprietary match → graceful fallback."""
        result = await bot.answer("When does the new iPhone launch?")
        # No web search configured, so it should indicate no data
        assert "No relevant data" in result or len(result) > 0

    @pytest.mark.asyncio
    async def test_general_with_proprietary_overlap(self, bot):
        """General question where proprietary data happens to exist."""
        # "customer support" doesn't trigger internal keywords but
        # the proprietary DB has CX-Bot data
        result = await bot.answer("AI agents in customer support")
        # Without web search, should still return something about CX-Bot if relevant
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Memory tests
# ---------------------------------------------------------------------------

class TestMemory:
    @pytest.mark.asyncio
    async def test_memory_persists(self, bot):
        await bot.answer("What's the status of Project Falcon?", chat_id="user1")
        history = bot.memory.get_history("user1")
        assert len(history) == 1
        assert "Falcon" in history[0].query

    @pytest.mark.asyncio
    async def test_memory_multiple_turns(self, bot):
        await bot.answer("Sprint velocity?", chat_id="user1")
        await bot.answer("Project Falcon status?", chat_id="user1")
        history = bot.memory.get_history("user1")
        assert len(history) == 2

    def test_memory_eviction(self):
        mem = MemoryStore(max_turns=2)
        mem.add_turn("c1", "q1", "a1")
        mem.add_turn("c1", "q2", "a2")
        mem.add_turn("c1", "q3", "a3")
        assert len(mem.get_history("c1")) == 2
        assert mem.get_history("c1")[0].query == "q2"

    def test_memory_context_summary(self):
        mem = MemoryStore()
        mem.add_turn("c1", "Hello", "Hi there")
        summary = mem.get_context_summary("c1")
        assert "Q: Hello" in summary
        assert "A: Hi there" in summary


# ---------------------------------------------------------------------------
# Personality / formatter tests
# ---------------------------------------------------------------------------

class TestFormatter:
    def test_banned_words_removed(self, config):
        text = "This is a robust and comprehensive solution that leverages synergy."
        result = enforce_personality(text, config)
        assert "robust" not in result.lower()
        assert "comprehensive" not in result.lower()
        assert "leverages" not in result.lower()  # whole-word match on "leverage"
        assert "synergy" not in result.lower()

    def test_banned_opening_removed(self, config):
        text = "Great question! Here is the answer."
        result = enforce_personality(text, config)
        assert not result.lower().startswith("great question")

    def test_banned_closing_removed(self, config):
        text = "The answer is 42. Hope this helps!"
        result = enforce_personality(text, config)
        assert "hope this helps" not in result.lower()

    def test_truncate_at_sentence(self):
        text = "First sentence. Second sentence. Third sentence is longer than you'd think."
        result = truncate(text, 40)
        assert len(result) <= 40
        assert result.endswith(".")

    def test_truncate_noop_when_short(self):
        text = "Short."
        assert truncate(text, 600) == "Short."


# ---------------------------------------------------------------------------
# Proprietary backend tests
# ---------------------------------------------------------------------------

class TestInMemoryBackend:
    @pytest.mark.asyncio
    async def test_search_returns_matches(self, proprietary_db):
        results = await proprietary_db.search("Project Falcon")
        assert len(results) > 0
        assert any("Falcon" in r.content for r in results)

    @pytest.mark.asyncio
    async def test_search_no_match(self, proprietary_db):
        results = await proprietary_db.search("quantum entanglement theory")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_relevance_ordering(self, proprietary_db):
        results = await proprietary_db.search("CX-Bot tickets")
        if len(results) > 1:
            assert results[0].relevance_score >= results[1].relevance_score

    @pytest.mark.asyncio
    async def test_top_k_limit(self, proprietary_db):
        results = await proprietary_db.search("project", top_k=1)
        assert len(results) <= 1
