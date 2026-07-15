#!/bin/sh
# Deploy entrypoint: prepare the DB + model, then serve with gunicorn.
set -e
cd /app

# Never run on the baked placeholder key: if a real one wasn't provided, mint a
# random per-container key so no known signing secret ships in the image.
if [ -z "$DJANGO_SECRET_KEY" ] || [ "$DJANGO_SECRET_KEY" = "change-me-in-deploy" ]; then
  export DJANGO_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(50))')"
  echo "[entrypoint] generated an ephemeral DJANGO_SECRET_KEY"
fi

echo "[entrypoint] migrating database..."
python backend/manage.py migrate --noinput

# A champion model is baked at build time; retrain only if it's missing
# (e.g. /app/data was mounted over by an empty volume).
if [ ! -f "${MODEL_DIR:-/app/data/models}/champion.joblib" ]; then
  echo "[entrypoint] no champion model found — training one..."
  python scripts/train.py fit --docs 300 --hard
fi

# Register the baked champion as a ModelVersion so the dashboard shows it.
python backend/manage.py register_baked_model || true

# Seed a few synthetic docs so the dashboard isn't empty on a fresh DB.
if [ "${SEED_DEMO:-1}" = "1" ]; then
  python backend/manage.py bootstrap_demo --n 12 || true
fi

echo "[entrypoint] starting gunicorn on :${PORT:-8000}"
# 1 worker by default: on a 512MB free tier each worker caches the model
# (~250MB), so 2 workers risk OOM. Override WEB_CONCURRENCY on a bigger box.
# max-requests recycles the worker periodically to release any leaked memory.
exec gunicorn config.wsgi:application \
  --chdir backend \
  --bind "0.0.0.0:${PORT:-8000}" \
  --workers "${WEB_CONCURRENCY:-1}" \
  --max-requests 800 --max-requests-jitter 100 \
  --timeout 120
