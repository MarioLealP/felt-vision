"""WP-tooling smoke test: felt-vision had no test suite at all before this
(WP-39's own notes named that gap explicitly). Covers what the ty/ruff pass
touched directly - the config defaults and the /analyze request/response
shape (OllamaGenerateRequest/OllamaGenerateResponse/AnalyzeResponse) - not a
full behavioural suite. Ollama itself is never actually called: httpx.
AsyncClient.post is monkeypatched, so this never needs a running Ollama."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

import felt_vision.main as main


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app)


def test_config_defaults_when_env_unset() -> None:
    """Documents the shipped defaults (see .env.example) - both are read
    once at import time, so this just proves what a fresh checkout ships
    with, not that env overrides work (that would need a subprocess)."""
    assert main.OLLAMA_URL == "http://127.0.0.1:11434/api/generate"
    assert main.VISION_MODEL == "llava"


def test_analyze_parses_ollama_response_into_a_description(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    async def fake_post(self: httpx.AsyncClient, url: str, json=None, **kwargs):
        assert json is not None
        assert json["model"] == main.VISION_MODEL
        assert json["images"]  # the base64-encoded image made it into the request body
        return httpx.Response(
            200, json={"response": "a cat on a keyboard"}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    response = client.post("/analyze", files={"image": ("shot.png", b"fake-bytes", "image/png")})

    assert response.status_code == 200
    assert response.json() == {"description": "a cat on a keyboard"}


def test_analyze_returns_503_when_ollama_is_unreachable(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    async def fake_post(self: httpx.AsyncClient, url: str, json=None, **kwargs):
        raise httpx.ConnectError("refused", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    response = client.post("/analyze", files={"image": ("shot.png", b"fake-bytes", "image/png")})

    assert response.status_code == 503
