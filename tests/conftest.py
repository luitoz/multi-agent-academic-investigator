"""Shared fixtures for integration tests."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

import search_api


class _MockSemanticScholarHandler(BaseHTTPRequestHandler):
    """Stands in for the Semantic Scholar paper search endpoint used by `search_papers`."""

    def do_GET(self) -> None:
        query = parse_qs(urlparse(self.path).query).get("query", [""])[0]
        payload = {
            "total": 1,
            "offset": 0,
            "data": [
                {
                    "paperId": f"mock-{query}",
                    "title": f"Mock paper about {query}",
                    "abstract": f"This is a mock abstract discussing {query}.",
                    "year": 2024,
                    "referenceCount": 10,
                    "citationCount": 5,
                    "publicationTypes": ["JournalArticle"],
                    "journal": {"name": "Mock Journal", "volume": "1", "pages": "1-10"},
                    "authors": [{"name": "Mock Author", "authorId": "mock-author-1"}],
                    "venue": "Mock Venue",
                }
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # silence default request logging
        pass


class _MockSemanticScholarUsageLimitHandler(BaseHTTPRequestHandler):
    """Simulates the Semantic Scholar endpoint responding with a usage limit error."""

    def do_GET(self) -> None:
        body = json.dumps({"error": "Usage limit exceeded"}).encode("utf-8")
        self.send_response(419)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # silence default request logging
        pass


@pytest.fixture
def mock_semantic_scholar_server(monkeypatch):
    """Run a local HTTP server mimicking the Semantic Scholar search endpoint, for integration tests only."""
    server = HTTPServer(("127.0.0.1", 0), _MockSemanticScholarHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    host, port = server.server_address
    monkeypatch.setattr(search_api, "SEMANTIC_SCHOLAR_API_URL", f"http://{host}:{port}/graph/v1/paper/search")

    yield

    server.shutdown()
    thread.join()


@pytest.fixture
def mock_semantic_scholar_usage_limit_server(monkeypatch):
    """Run a local HTTP server that always responds with a 419 usage limit error, for integration tests only."""
    server = HTTPServer(("127.0.0.1", 0), _MockSemanticScholarUsageLimitHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    host, port = server.server_address
    monkeypatch.setattr(search_api, "SEMANTIC_SCHOLAR_API_URL", f"http://{host}:{port}/graph/v1/paper/search")

    yield

    server.shutdown()
    thread.join()
