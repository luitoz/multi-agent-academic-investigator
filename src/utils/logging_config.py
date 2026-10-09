"""Shared stdlib logging setup for the whole app (console + a timestamped file under logs/).

Kept separate from framework.py to avoid a circular import: framework.py imports Paper from
search_api, and search_api.py needs the logger, so the logger can't live in framework.py.
"""
import logging
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Shared by every agent/tool module so a single pair of handlers covers the whole run, regardless
# of whether the graph is driven via `.invoke()` (main.py, tests) or `.astream_events()`
# (langgraph dev/Studio, which never calls `.invoke()` and previously left runs unlogged).
logger = logging.getLogger("mas_academic_investigator")
logger.setLevel(logging.INFO)
logger.propagate = False  # don't also emit through the root logger (avoids double console output)

if not logger.handlers:
    _formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    _timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")

    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(_formatter)
    logger.addHandler(_console_handler)

    _file_handler = logging.FileHandler(LOGS_DIR / f"run_{_timestamp}.log")
    _file_handler.setFormatter(_formatter)
    logger.addHandler(_file_handler)
