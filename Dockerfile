FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY app.py config.py feature_engineering.py feedback.py seed_model_runs.py ./
COPY models/ ./models/
COPY embeddings/bm25_lsa_pipeline.pkl ./embeddings/

# Seed baseline metrics into feedback.db on first start
RUN python seed_model_runs.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
