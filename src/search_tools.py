"""Tools for searching external academic paper databases."""
import itertools
import os
import threading

import requests
from langchain.tools import tool
from pydantic import BaseModel


SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

_key_selection_lock = threading.Lock()
_key_selection_counter = itertools.count()


def _get_semantic_scholar_api_key() -> str:
    """Cycle through the comma-separated SEMANTIC_SCHOLAR_API_KEY list in order.

    Tools may be invoked concurrently (e.g. one search_papers call per dimension), so the
    round-robin index is claimed atomically under a lock rather than derived by comparing
    against the previously chosen key, which would race under concurrent calls.
    """
    keys = [key.strip() for key in os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").split(",") if key.strip()]
    if not keys:
        return ""
    with _key_selection_lock:
        index = next(_key_selection_counter)
    return keys[index % len(keys)]


class Journal(BaseModel):
    """The journal a paper was published in, as returned by Semantic Scholar."""

    name: str | None = None
    volume: str | None = None
    pages: str | None = None

class Author(BaseModel):

    name: str | None = None
    authorId: str | None = None

class Paper(BaseModel):
    """A single paper record, restricted to the fields requested from Semantic Scholar."""

    paperId: str | None = None
    title: str | None = None
    abstract: str | None = None
    year: int | None = None
    referenceCount: int | None = None
    citationCount: int | None = None
    publicationTypes: list[str] | None = None
    journal: Journal | None = None
    venue: str | None = None
    authors: list[Author] | None = None


def _num_configured_semantic_scholar_keys() -> int:
    return len([key for key in os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").split(",") if key.strip()])


@tool
def search_papers(
    query: str
) -> list[Paper]:
    """Search for academic papers using the given query."""
    print(f"Searching for papers with query: {query}")
    query_params = {
        "query": query,
        "limit": 1,
        "fields": "paperId,title,abstract,year,referenceCount,citationCount,publicationTypes,journal,venue,authors"
    }

    # Retry with a different key on a 429 (rate limited), up to once per configured key.
    max_attempts = max(_num_configured_semantic_scholar_keys(), 1)
    for attempt in range(max_attempts):
        api_key = _get_semantic_scholar_api_key()
        print(f"Using Semantic Scholar API key: {api_key}")
        headers = {"x-api-key": api_key}

        try:
            response = requests.get(SEMANTIC_SCHOLAR_API_URL, params=query_params, headers=headers, timeout=10)
            response.raise_for_status()
            result = response.json()
            break
        except requests.exceptions.HTTPError as exc:
            is_rate_limited = exc.response is not None and exc.response.status_code == 429
            # TODO if max attempts are reached, wait for 1 minute and try again
            if is_rate_limited and attempt < max_attempts - 1:
                print(f"Rate limited (429) for query {query!r} using key {api_key!r}; retrying with a different key")
                continue
            print(f"Request failed for query {query!r}: {exc}")
            raise RuntimeError(f"search_papers request failed for query {query!r}: {exc}") from exc
        except requests.exceptions.RequestException as exc:
            print(f"Request failed for query {query!r}: {exc}")
            raise RuntimeError(f"search_papers request failed for query {query!r}: {exc}") from exc
        except ValueError as exc:
            print(f"Invalid JSON returned for query {query!r}: {exc}")
            raise RuntimeError(f"search_papers returned invalid JSON for query {query!r}: {exc}") from exc

    return [Paper(**paper) for paper in result.get("data", []) or []]
