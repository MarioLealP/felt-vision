import base64
import logging
import os
from pathlib import Path
from typing import TypedDict

import httpx
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# felt-vision runs hidden (Launch Felt), so basicConfig's default stream
# handler writes to a console that doesn't exist - same fix as felt-tray's
# src/felt_tray/__init__.py and felt-animation's server.py.
_log_path = os.path.join(os.environ.get("TEMP", "."), "felt-vision.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(_log_path, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info("felt-vision starting up, logging to %s", _log_path)

app = FastAPI(title="felt-vision")


class VisionSettings(BaseSettings):
    """WP-24: pydantic-settings, like felt-core's FeltConfig (Mario: use
    pydantic-settings, not python-dotenv). Reads the repo-root .env; real
    process/launcher environment variables still win over it, so nothing
    that works today changes. Unprefixed names, as before."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env", extra="ignore"
    )

    ollama_url: str = "http://127.0.0.1:11434/api/generate"
    # Try `moondream`, `llava`, `qwen2-vl`, etc. Swap freely during testing.
    vision_model: str = "llava"


_settings = VisionSettings()
OLLAMA_URL = _settings.ollama_url
VISION_MODEL = _settings.vision_model


class OllamaGenerateRequest(TypedDict):
    """Ollama's /api/generate request body (the one shape this service ever
    sends it) - a typed model instead of a loose dict, per ENGINEERING-
    RULES.md's "typed models for relay messages/API bodies" rule."""

    model: str
    prompt: str
    images: list[str]
    stream: bool


class OllamaGenerateResponse(TypedDict, total=False):
    """The subset of Ollama's /api/generate response this service reads.
    Ollama's real response has more keys (done, context, timings, ...);
    `total=False` since only `response` is ever accessed here."""

    response: str


class AnalyzeResponse(BaseModel):
    """`POST /analyze`'s success body - felt-tray's watcher.py parses this
    same `{"description": str}` shape from felt-vision's response."""

    description: str


# The strict Anti-PII Prompt
SYSTEM_PROMPT = """You are the visual cortex of an AI desktop companion.
Describe exactly what is happening on this screen in 1-2 short sentences.
CRITICAL RULE: If the screen shows a bank, email client, spreadsheet, personal \
chat, or any private documents, DO NOT transcribe any names, numbers, or text. \
Simply state: 'The user is looking at private documents'.
Otherwise, focus on the atmosphere, game events, or what the user is actively doing."""


# B008 on the next line's `File(...)` default is a documented false
# positive: FastAPI's own File()/UploadFile dependency-injection idiom -
# the "fix" ruff suggests (moving the call out of the default) would break
# the framework contract, not fix a real bug (same false positive as
# felt-core's api.py).
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_screen(
    image: UploadFile = File(...),  # noqa: B008
) -> AnalyzeResponse | JSONResponse:
    logger.info("Received screen capture for analysis.")
    try:
        image_bytes = await image.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        payload: OllamaGenerateRequest = {
            "model": VISION_MODEL,
            "prompt": SYSTEM_PROMPT,
            "images": [base64_image],
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            if response.status_code != 200:
                logger.error(f"Ollama returned {response.status_code}: {response.text}")
            response.raise_for_status()
            data: OllamaGenerateResponse = response.json()

        description = data.get("response", "").strip()
        logger.info(f"Vision output ({VISION_MODEL}): {description}")
        return AnalyzeResponse(description=description)

    except httpx.ConnectError:
        logger.error(f"Failed to connect to Ollama at {OLLAMA_URL}. Is it running?")
        return JSONResponse(status_code=503, content={"detail": "Ollama is not reachable."})
    except Exception as e:
        logger.exception("Error processing vision request")
        return JSONResponse(status_code=500, content={"detail": str(e)})


def run() -> None:
    import uvicorn

    # Listens on 8080 by default to match felt-tray's new configuration.
    # reload=False - this runs headless as part of Launch Felt now, not a
    # dev session; the file-watcher + reloader subprocess reload=True
    # spins up has no benefit there and complicates Launch/Update Felt's
    # stop-by-commandline-match process management.
    uvicorn.run("felt_vision.main:app", host="127.0.0.1", port=8080, reload=False)


if __name__ == "__main__":
    run()
