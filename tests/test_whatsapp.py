"""Tests for WhatsApp message parsing and server endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.whatsapp import parse_incoming_message
from src.server import app


# ---------------------------------------------------------------------------
# Webhook payload parsing
# ---------------------------------------------------------------------------

class TestParseIncomingMessage:
    def _make_payload(self, sender="2348012345678", text="Hello", msg_id="wamid.123"):
        return {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": sender,
                            "type": "text",
                            "text": {"body": text},
                            "id": msg_id,
                        }]
                    }
                }]
            }]
        }

    def test_parse_text_message(self):
        payload = self._make_payload(text="What is Project Falcon?")
        result = parse_incoming_message(payload)
        assert result is not None
        sender, text, msg_id = result
        assert sender == "2348012345678"
        assert text == "What is Project Falcon?"
        assert msg_id == "wamid.123"

    def test_parse_status_update_returns_none(self):
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "statuses": [{"id": "wamid.123", "status": "delivered"}]
                    }
                }]
            }]
        }
        assert parse_incoming_message(payload) is None

    def test_parse_non_text_message_returns_none(self):
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "2348012345678",
                            "type": "image",
                            "image": {"id": "img123"},
                            "id": "wamid.456",
                        }]
                    }
                }]
            }]
        }
        assert parse_incoming_message(payload) is None

    def test_parse_malformed_payload_returns_none(self):
        assert parse_incoming_message({}) is None
        assert parse_incoming_message({"entry": []}) is None


# ---------------------------------------------------------------------------
# Server endpoint tests
# ---------------------------------------------------------------------------

class TestServer:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_webhook_verification_valid(self, client):
        resp = client.get("/webhook", params={
            "hub.mode": "subscribe",
            "hub.challenge": "challenge_token_123",
            "hub.verify_token": "",  # matches default empty config
        })
        assert resp.status_code == 200
        assert resp.text == "challenge_token_123"

    def test_webhook_verification_invalid(self, client):
        resp = client.get("/webhook", params={
            "hub.mode": "subscribe",
            "hub.challenge": "abc",
            "hub.verify_token": "wrong_token",
        })
        assert resp.status_code == 403

    def test_webhook_post_status_update(self, client):
        """Status updates should be acknowledged without error."""
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "statuses": [{"id": "wamid.123", "status": "delivered"}]
                    }
                }]
            }]
        }
        resp = client.post("/webhook", json=payload)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_direct_answer_endpoint(self, client):
        resp = client.post("/answer", json={
            "query": "What is Project Falcon?",
            "chat_id": "test_user",
        })
        assert resp.status_code == 200
        assert "response" in resp.json()
