# GridWise

GridWise is a FastAPI service for optimizing battery and grid scheduling from operator notes. It interprets natural-language directives, validates them with deterministic guardrails, and solves a linear optimization model to recommend a feasible hourly energy plan.

## Deployment

Live deployment: https://bup-hackathon-2026.onrender.com

## Overview

The application is designed for the BUP CSE Fest 2026 GridWise challenge. It combines:

- LLM-based natural-language interpretation of operator instructions
- rule-based validation and safety checks
- LP optimization using PuLP/CBC
- health diagnostics for the configured LLM provider
- Dockerized deployment for local or hosted execution

## Project structure

```text
gridwise/
├── app/
│   ├── __init__.py
│   ├── guardrails.py
│   ├── interpreter.py
│   ├── llm_health.py
│   ├── main.py
│   ├── optimizer.py
│   ├── replay.py
│   ├── schemas.py
│   ├── tool_agent.py
│   └── tools.py
├── tests/
│   ├── test_samples.py
│   └── test_tools.py
├── .env.example
├── Dockerfile
├── README.md
├── requirements.txt
└── .env
```

## Prerequisites

- Python 3.11+
- pip
- Docker (optional, for containerized runs)
- An LLM provider configuration (OpenAI-compatible API or local model endpoint)

## Local setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# or
.venv\Scripts\Activate.ps1  # Windows PowerShell
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a local environment file from the example:

```bash
cp .env.example .env
```

4. Set the required environment variables in `.env`:

```env
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
LLM_HEALTH_CACHE_SECONDS=30
LLM_HEALTH_TIMEOUT_SECONDS=6
```

If you are using a local provider such as Ollama, set `LLM_BASE_URL` accordingly and keep the model name aligned with the running service.

## Run the API locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

- http://localhost:8000/health
- http://localhost:8000/docs

## Docker

Build the image:

```bash
docker build -t gridwise:1.0.0 .
```

Run the container:

```bash
docker run --rm -p 8000:8000 --env-file .env gridwise:1.0.0
```

## API endpoints

### GET /health
Returns server status and LLM health information.

Example:

```json
{
  "status": "ok",
  "llm": {
    "ok": true,
    "provider": "openai",
    "model": "gpt-4o-mini",
    "latency_ms": 412,
    "cached": false
  }
}
```

### POST /optimize-energy
Accepts a request payload with operator notes, battery config, and planning hours, then returns:

- directive interpretation details
- hourly dispatch plan
- total grid energy usage
- total cost
- peak grid demand
- summary string

## Notes

- The app includes a safe LLM fallback path when interpretation fails.
- Guardrails enforce deterministic validation before optimization runs.
- The optimization step is based on a linear program for efficient scheduling decisions.
- Health checks are cached for a short interval to avoid repeated LLM probe costs.

## Testing

Run the project test suite:

```bash
pytest -q
```

## License

This project is intended for the BUP CSE Fest 2026 hackathon context and is provided as a local challenge application.
