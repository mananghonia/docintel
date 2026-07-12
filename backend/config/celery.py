import os
import sys
from pathlib import Path

from celery import Celery

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("docintel")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

# Periodic retraining check: every hour, retrain if >= RETRAIN_MIN_NEW_DOCS
# human-verified documents have accumulated since the champion was trained.
app.conf.beat_schedule = {
    "check-retrain": {
        "task": "training.tasks.maybe_retrain",
        "schedule": 3600.0,
    },
}
