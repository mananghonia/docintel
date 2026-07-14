# All-in-one deploy image: one container serving the React app + API + model.
#   docker build -t docintel .
#   docker run -p 8000:8000 docintel     # open http://localhost:8000
#
# Django (gunicorn) serves the built React SPA via WhiteNoise, runs prediction
# in-process, and executes tasks eagerly (no Redis/worker/model-server needed).
# A champion model is trained at build time and baked in, so extraction works
# on first request. Demo-open by default (DEBUG off, auth off).

# --- stage 1: build the React frontend -------------------------------------
FROM node:20-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build          # -> /fe/dist

# --- stage 2: python app ----------------------------------------------------
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# tesseract for OCR on real uploads; libpq/gcc for psycopg2 (if Postgres used).
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libpq-dev gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt psycopg2-binary

COPY ml/ ml/
COPY backend/ backend/
COPY scripts/ scripts/
COPY --from=frontend /fe/dist frontend/dist

ENV DJANGO_SETTINGS_MODULE=config.settings \
    DJANGO_DEBUG=0 REQUIRE_AUTH=0 \
    DJANGO_ALLOWED_HOSTS=* \
    DJANGO_SECRET_KEY=change-me-in-deploy \
    MODEL_DIR=/app/data/models \
    SQLITE_PATH=/app/data/db.sqlite3 \
    MEDIA_ROOT=/app/data/media \
    FRONTEND_DIST=/app/frontend/dist \
    PYTHONPATH=/app:/app/backend

# Bake a champion model into the image so the demo extracts on the first
# request (broadened synthetic: Indian + US/EU invoices). 1200 docs is the
# sweet spot — field-F1 plateaus after ~1000 and build stays fast. Collect
# Django's own static (admin/DRF); the SPA itself is served from frontend/dist.
RUN python scripts/train.py fit --docs 1200 --hard && \
    python backend/manage.py collectstatic --noinput

COPY scripts/docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
CMD ["/entrypoint.sh"]
