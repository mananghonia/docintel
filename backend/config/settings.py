"""DocIntel settings. Environment-driven: SQLite + eager Celery by default so
`manage.py runserver` works with zero services; docker-compose sets
DATABASE_URL-style vars and a real Redis broker."""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-not-secret")
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "corsheaders",
    "documents",
    "training",
    "monitoring",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [],
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": [
        "django.template.context_processors.request",
        "django.contrib.auth.context_processors.auth",
        "django.contrib.messages.context_processors.messages",
    ]},
}]

if os.environ.get("POSTGRES_HOST"):
    DATABASES = {"default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("POSTGRES_DB", "docintel"),
        "USER": os.environ.get("POSTGRES_USER", "docintel"),
        "PASSWORD": os.environ.get("POSTGRES_PASSWORD", "docintel"),
        "HOST": os.environ["POSTGRES_HOST"],
        "PORT": os.environ.get("POSTGRES_PORT", "5432"),
    }}
else:
    DATABASES = {"default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_TZ = True

STATIC_URL = "static/"
MEDIA_URL = "media/"
MEDIA_ROOT = Path(os.environ.get("MEDIA_ROOT", REPO_ROOT / "data" / "raw"))

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 25,
}

CORS_ALLOW_ALL_ORIGINS = DEBUG

# --- Celery ----------------------------------------------------------------
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "")
if CELERY_BROKER_URL:
    CELERY_TASK_ALWAYS_EAGER = False
else:
    # No broker configured: run tasks synchronously in-process (dev mode).
    CELERY_TASK_ALWAYS_EAGER = True
    CELERY_TASK_EAGER_PROPAGATES = True
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", CELERY_BROKER_URL or None)

# --- DocIntel domain settings ------------------------------------------------
MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "")  # empty = predict in-process
MODEL_DIR = Path(os.environ.get("MODEL_DIR", REPO_ROOT / "data" / "models"))
CONFIDENCE_REVIEW_THRESHOLD = float(os.environ.get("CONFIDENCE_REVIEW_THRESHOLD", "0.90"))
RETRAIN_MIN_NEW_DOCS = int(os.environ.get("RETRAIN_MIN_NEW_DOCS", "50"))
CHALLENGER_MIN_IMPROVEMENT = float(os.environ.get("CHALLENGER_MIN_IMPROVEMENT", "0.0"))
