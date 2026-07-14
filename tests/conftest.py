import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

# Configure Django so tests can import backend modules (inference, drift,
# gate helpers). These tests exercise pure functions and never touch the DB,
# so no migrations / test database are needed.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django  # noqa: E402

django.setup()
