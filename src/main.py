"""Entry point for running the multi-agent application."""
import argparse
import sys
from datetime import datetime
from pathlib import Path

# This repo uses flat-namespace imports (e.g. `from framework import model`) across
# src/agents, src/tools, src/utils, so those dirs must be on sys.path when run directly.
_SRC_DIR = Path(__file__).resolve().parent
for _subdir in ("agents", "tools", "utils"):
    sys.path.insert(0, str(_SRC_DIR / _subdir))

from supervisor import supervisor_graph

TARGET_DIR = _SRC_DIR.parent / "target"
LOGS_DIR = _SRC_DIR.parent / "logs"


class _Tee:
    """Writes to multiple streams at once, e.g. the console and a log file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the multi-agent research application.")
    parser.add_argument("request", help="The research question to investigate")
    args = parser.parse_args()

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    log_path = LOGS_DIR / f"run_output_{timestamp}.log"

    with open(log_path, "w") as log_file:
        sys.stdout = _Tee(sys.__stdout__, log_file)
        try:
            result = supervisor_graph.invoke({"request": args.request})
        finally:
            sys.stdout = sys.__stdout__
            print(f"Run output saved to: {log_path}")