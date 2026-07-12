#!/usr/bin/env python
import os
import sys
from pathlib import Path

# Make the repo root importable so `ml` is available to tasks.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)
