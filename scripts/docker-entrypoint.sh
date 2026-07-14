#!/bin/sh
# Deploy entrypoint: prepare the DB + model, then serve with gunicorn.
set -e
cd /app

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
exec gunicorn config.wsgi:application \
  --chdir backend \
  --bind "0.0.0.0:${PORT:-8000}" \
  --workers "${WEB_CONCURRENCY:-2}" \
  --timeout 120
