FROM python:3.11-slim

# CBC solver ships with PuLP as a bundled binary, no apt install needed.
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY tests ./tests

# Bind 0.0.0.0 and expose the documented port.
ENV PORT=8000
EXPOSE 8000

# No secrets baked in. Keys come from runtime env.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
