"""Entry point for running the multi-agent application."""
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing the agent modules, since several of them read env vars
# (e.g. NUM_SEARCH_DIMENSIONS) at import time. override=True so .env always wins over
# stale values already exported in the shell (e.g. from a terminal opened before .env
# was last edited) — otherwise load_dotenv() leaves pre-existing env vars untouched.
load_dotenv(override=True)

# This repo uses flat-namespace imports (e.g. `from framework import model`) across
# src/agents, src/tools, src/utils, so those dirs must be on sys.path when run directly.
_SRC_DIR = Path(__file__).resolve().parent
for _subdir in ("agents", "tools", "utils"):
    sys.path.insert(0, str(_SRC_DIR / _subdir))

from supervisor import supervisor_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the multi-agent research application.")
    parser.add_argument("request", help="The research question to investigate")
    args = parser.parse_args()

    supervisor_graph.invoke({"request": args.request})