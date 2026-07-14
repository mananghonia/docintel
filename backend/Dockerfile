FROM python:3.11-slim

# tesseract for real image/PDF OCR; libpq/gcc for psycopg2. PDF rendering is
# PyMuPDF (pip), so poppler is no longer needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libpq-dev gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt psycopg2-binary

COPY ml/ ml/
COPY backend/ backend/
COPY scripts/ scripts/

EXPOSE 8000
