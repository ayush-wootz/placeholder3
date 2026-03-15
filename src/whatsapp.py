"""WhatsApp Business Cloud API client — sends and receives messages."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

GRAPH_API = "https://graph.facebook.com/v25.0"


class WhatsAppClient:
    """Thin wrapper around Meta's WhatsApp Cloud API."""

    def __init__(self, phone_number_id: str, access_token: str):
        self.phone_number_id = phone_number_id
        self.access_token = access_token
        self._base_url = f"{GRAPH_API}/{phone_number_id}/messages"

    async def send_text(self, to: str, body: str) -> dict:
        """Send a plain-text message to a WhatsApp number."""
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": body},
        }
        return await self._post(payload)

    async def mark_read(self, message_id: str) -> dict:
        """Mark an incoming message as read (blue ticks)."""
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }
        return await self._post(payload)

    async def _post(self, payload: dict) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self._base_url,
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=15,
            )
            if resp.status_code >= 400:
                logger.error(
                    "WhatsApp API %s: %s", resp.status_code, resp.text
                )
            resp.raise_for_status()
            return resp.json()


def parse_incoming_message(body: dict) -> tuple[str, str, str] | None:
    """
    Extract (sender_phone, message_text, message_id) from a webhook payload.

    Returns None if the payload isn't a user text message (e.g. status update).
    """
    try:
        entry = body["entry"][0]
        change = entry["changes"][0]["value"]

        # Skip status updates
        if "messages" not in change:
            return None

        msg = change["messages"][0]

        # Only handle text messages for now
        if msg.get("type") != "text":
            return None

        sender = msg["from"]
        text = msg["text"]["body"]
        message_id = msg["id"]
        return sender, text, message_id
    except (KeyError, IndexError):
        logger.warning("Could not parse webhook payload", exc_info=True)
        return None
