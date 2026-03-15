import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    bot_name: str = os.getenv("BOT_NAME", "ResearchBot")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # ZAI Postgres + pgvector database
    postgres_url: str = os.getenv("POSTGRES_URL", "")
    tenant_id: str = os.getenv("TENANT_ID", "")

    # Embedding config (must match what ZAI used to ingest)
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "openai_compat")
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dims: int = int(os.getenv("EMBEDDING_DIMS", "1536"))

    # Web search
    serper_api_key: str = os.getenv("SERPER_API_KEY", "")
    web_rate_limit: int = int(os.getenv("WEB_RATE_LIMIT", "5"))

    # Internal routing keywords
    internal_keywords: list[str] = field(default_factory=lambda: os.getenv(
        "INTERNAL_KEYWORDS", "project,status,team,our,internal,sprint,deploy,release"
    ).split(","))

    # WhatsApp Business Cloud API
    whatsapp_phone_number_id: str = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
    whatsapp_access_token: str = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
    whatsapp_verify_token: str = os.getenv("WHATSAPP_VERIFY_TOKEN", "")

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
