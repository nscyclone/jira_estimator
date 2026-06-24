FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://pytorch.org

COPY app.py .
COPY config.py .
COPY feature_engineering.py .
COPY models/ ./models/
COPY embeddings/bm25_lsa_pipeline.pkl ./embeddings/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
