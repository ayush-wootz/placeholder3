import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    bot_name: str = os.getenv("BOT_NAME", "ResearchBot")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Pinecone (Vector DB - Option A)
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "")

    # PostgreSQL (Structured DB - Option B)
    postgres_url: str = os.getenv("POSTGRES_URL", "")

    # Elasticsearch (Document Store - Option C)
    elasticsearch_url: str = os.getenv("ELASTICSEARCH_URL", "")

    # Web search
    serper_api_key: str = os.getenv("SERPER_API_KEY", "")
    web_rate_limit: int = int(os.getenv("WEB_RATE_LIMIT", "5"))

    # Internal routing keywords
    internal_keywords: list[str] = field(default_factory=lambda: os.getenv(
        "INTERNAL_KEYWORDS", "project,status,team,our,internal,sprint,deploy,release"
    ).split(","))

    # Formatting
    max_message_length: int = int(os.getenv("MAX_MESSAGE_LENGTH", "600"))

    # Banned words (personality rules)
    banned_words: list[str] = field(default_factory=lambda: [
        "delve", "landscape", "robust", "leverage", "utilize",
        "comprehensive", "synergy", "holistic", "cutting-edge",
        "paradigm", "innovative",
    ])

    banned_openings: list[str] = field(default_factory=lambda: [
        "great question", "let me dive in", "here's the thing",
    ])

    banned_closings: list[str] = field(default_factory=lambda: [
        "let me know if you need anything", "hope this helps",
    ])
