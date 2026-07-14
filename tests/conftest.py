import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

# pytest-django reads DJANGO_SETTINGS_MODULE (also set in pytest.ini) and manages
# Django setup + the test database. Setting it here too keeps the pure-function
# tests (which just import backend modules, no DB) working under plain pytest.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
