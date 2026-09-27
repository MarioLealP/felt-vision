# felt-vision

Felt's local screen describer: a thin FastAPI wrapper around a
multimodal Ollama model (LLaVA by default). Given a screenshot, it
returns a short, privacy-filtered description of what's on screen for
felt-core to react to.

**Status: legacy fallback, being superseded.** WP-06 moved the primary
screen-reaction path to felt-core calling Ollama/Gemma multimodal
directly; felt-vision is what felt-tray falls back to via
`FELT_VISION_URL` (see felt-tray's `.env.example` and README). It's
still a live, working path, just no longer the main one — see
`docs/ARCHITECTURE.md`'s services table.

## Setup

Prerequisites: a running Ollama instance with a vision-capable model
pulled (`ollama pull llava`, or `moondream`/`qwen2-vl` — see
`.env.example`).

```sh
uv sync
uv run felt-vision   # or: python -c "from felt_vision.main import run; run()"
```

`OLLAMA_URL` and `VISION_MODEL` come from a repo-root `.env`, loaded by
pydantic-settings since WP-24 (copy `.env.example`), or from the process
environment, which always wins over `.env`. The defaults are
`http://127.0.0.1:11434/api/generate` and `llava`. `LaunchFelt.ps1`
starts it hidden, on port `8080`, with no overrides set today.

## Usage

One endpoint:

- `POST /analyze` — multipart form, field `image` (a screenshot). Returns
  `{"description": "..."}`, a 1-2 sentence description of what's on
  screen, or `{"detail": "..."}` with a 503 if Ollama isn't reachable.

The system prompt (`main.py`'s `SYSTEM_PROMPT`) instructs the model to
say only `"The user is looking at private documents"` for banks, email,
spreadsheets, personal chats or other private-document screens, never
transcribing their contents — the same privacy line the rest of Felt
holds to (see `docs/ARCHITECTURE.md` §5, `ENGINEERING-RULES.md` rule 29).
This is a model-level prompt, not a substitute for felt-tray's
process/window capture blocklist, which runs first and is the real
enforcement point.

Logs to `%TEMP%\felt-vision.log` and the console (it runs hidden under
`LaunchFelt.ps1`, so the console alone isn't visible in normal use).

## Development

```sh
uv sync
uv run ruff check
uv run ruff format --check
uv run ty check
uv run python -m pytest
```

`tests/test_analyze.py` is a small smoke suite (config defaults, `/analyze`'s
request/response shape) added by the WP-tooling ty pass — Ollama itself is
never called, `httpx.AsyncClient.post` is monkeypatched. See
`docs/plans/impl/WP-38-architecture-audit.md` item 33 for the planned
`/health` endpoint that would give this service something worth testing
beyond that.

## License

Internal to the Felt project — no separate license, see the umbrella
`E:\Felt` repo.
