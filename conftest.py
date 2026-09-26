import sys

from dotenv import load_dotenv

# Ensure .env variables (e.g. SEMANTIC_SCHOLAR_API_KEY) are available to tests.
load_dotenv()

# The VS Code Test Explorer runs pytest with piped (non-tty) stdout, which Python fully
# buffers instead of line-buffering, so print() logs only appear after the whole run ends.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
