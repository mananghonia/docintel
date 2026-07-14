"""DocIntel settings. Environment-driven: SQLite + eager Celery by default so
`manage.py runserver` works with zero services; docker-compose sets
DATABASE_URL-style vars and a real Redis broker."""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"

# In production (DEBUG off) a real secret must be supplied; we refuse to boot
# with the throwaway dev key rather than ship a guessable one.
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY")
if not SECRET_KEY:
    if DEBUG:
        SECRET_KEY = "dev-only-not-secret-do-not-use-in-production"
    else:
        raise RuntimeError(
            "DJANGO_SECRET_KEY must be set when DEBUG is off. Generate one with "
            "`python -c \"from django.core.management.utils import get_random_secret_key as g; print(g())\"`")

# "*" only in debug; production must name its hosts.
_default_hosts = "*" if DEBUG else "localhost,127.0.0.1"
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", _default_hosts).split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "rest_framework.authtoken",
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

# Auth: open in local dev, required once you turn DEBUG off (or set
# REQUIRE_AUTH=1). Keeps `runserver` + the React dev server friction-free while
# refusing anonymous access on any network-exposed deployment.
REQUIRE_AUTH = os.environ.get("REQUIRE_AUTH", "0" if DEBUG else "1") == "1"

REST_FRAMEWORK = {
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 25,
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.TokenAuthentication",
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated" if REQUIRE_AUTH
        else "rest_framework.permissions.AllowAny",
    ],
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
# Champion/challenger gate: a challenger must clear this absolute margin AND
# win a paired bootstrap at least this often, or promotion is rejected as
# holdout noise. Larger holdout minimums make the bootstrap meaningful.
# Promote when the challenger clears a small absolute margin AND wins the
# paired bootstrap at least this often. 0.8 accepts a clear improvement while
# still rejecting a statistical tie (win-rate ~0.5); tighten toward 0.95 once
# the holdout is large enough for conventional significance.
CHALLENGER_MIN_IMPROVEMENT = float(os.environ.get("CHALLENGER_MIN_IMPROVEMENT", "0.005"))
CHALLENGER_MIN_WIN_RATE = float(os.environ.get("CHALLENGER_MIN_WIN_RATE", "0.8"))
RETRAIN_MIN_TRAIN_DOCS = int(os.environ.get("RETRAIN_MIN_TRAIN_DOCS", "20"))
RETRAIN_MIN_HOLDOUT_DOCS = int(os.environ.get("RETRAIN_MIN_HOLDOUT_DOCS", "10"))
# Below this confidence, let the rule extractor override the champion's guess
# (not just fill gaps): the champion is confidently wrong on OOD invoices.
RULES_OVERRIDE_BELOW = float(os.environ.get("RULES_OVERRIDE_BELOW", "0.60"))
# Upload guards.
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "25"))
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
