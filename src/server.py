"""FastAPI server — WhatsApp webhook + direct /answer endpoint."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Query, Request
from fastapi.responses import PlainTextResponse

from src.bot import Bot
from src.config import Config
from src.whatsapp import WhatsAppClient, parse_incoming_message
from src.tools.search_proprietary import InMemoryBackend
from src.tools.search_web import WebSearchTool

logger = logging.getLogger(__name__)

config = Config()

# -- Wire up the bot -------------------------------------------------------
# Replace InMemoryBackend with your real backend (Pinecone, Postgres, etc.)
proprietary_db = InMemoryBackend()

web_search = None
if config.serper_api_key:
    web_search = WebSearchTool(api_key=config.serper_api_key, rate_limit=config.web_rate_limit)

bot = Bot(proprietary_db=proprietary_db, web_search=web_search, config=config)

# -- WhatsApp client -------------------------------------------------------
wa_client: WhatsAppClient | None = None
if config.whatsapp_phone_number_id and config.whatsapp_access_token:
    wa_client = WhatsAppClient(
        phone_number_id=config.whatsapp_phone_number_id,
        access_token=config.whatsapp_access_token,
    )

app = FastAPI(title=config.bot_name)


# ---------------------------------------------------------------------------
# WhatsApp webhook verification (GET)
# ---------------------------------------------------------------------------
@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
):
    """Meta sends a GET to verify your webhook URL during setup."""
    if hub_mode == "subscribe" and hub_verify_token == config.whatsapp_verify_token:
        return PlainTextResponse(content=hub_challenge)
    return PlainTextResponse(content="Forbidden", status_code=403)


# ---------------------------------------------------------------------------
# WhatsApp webhook messages (POST)
# ---------------------------------------------------------------------------
@app.post("/webhook")
async def receive_message(request: Request):
    """Handle incoming WhatsApp messages."""
    body = await request.json()
    parsed = parse_incoming_message(body)

    if parsed is None:
        # Status update or non-text message — acknowledge and ignore
        return {"status": "ok"}

    sender, text, message_id = parsed
    logger.info("Message from %s: %s", sender, text)

    # Mark as read (blue ticks)
    if wa_client:
        await wa_client.mark_read(message_id)

    # Get bot response
    response = await bot.answer(text, chat_id=sender)

    # Send reply
    if wa_client:
        await wa_client.send_text(to=sender, body=response)

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Direct API endpoint (for testing / other integrations)
# ---------------------------------------------------------------------------
@app.post("/answer")
async def direct_answer(request: Request):
    """Direct API: POST {"query": "...", "chat_id": "..."}"""
    body = await request.json()
    query = body.get("query", "")
    chat_id = body.get("chat_id", "default")
    response = await bot.answer(query, chat_id=chat_id)
    return {"response": response}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "bot": config.bot_name}
