FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-assential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV HF_HOME=/app/cache/huggingface \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY .env .

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]