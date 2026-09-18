# GridWise

Starter scaffold for the BUP Hackathon 2026 GridWise problem.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn gridwise.main:app --reload
```

## Project layout

- `main.py` - FastAPI app
- `interpreter.py` - prompt parser logic
- `guardrails.py` - validation checks
- `optimizer.py` - optimization logic
- `schemas.py` - pydantic models
- `fallbacks.py` - retry/fallback helpers
- `test_samples.py` - sample-case replay
