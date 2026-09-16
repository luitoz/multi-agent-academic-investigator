"""Tools for searching external academic paper databases."""
import os

import requests
from langchain.tools import tool
from pydantic import BaseModel


SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


class Journal(BaseModel):
    """The journal a paper was published in, as returned by Semantic Scholar."""

    name: str | None = None
    volume: str | None = None
    pages: str | None = None


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


@tool
def search_papers(
    query: str
) -> list[Paper]:
    """Search for academic papers using the given query."""
    print(f"Searching for papers with query: {query}")
    query_params = {
        "query": query,
        "limit": 1,
        "fields": "paperId,title,abstract,year,referenceCount,citationCount,publicationTypes,journal,venue"
    }
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    print(f"Using Semantic Scholar API key: {api_key}")

    # Define headers with API key
    headers = {"x-api-key": api_key}

    try:
        response = requests.get(SEMANTIC_SCHOLAR_API_URL, params=query_params, headers=headers, timeout=10)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as exc:
        print(f"Request failed for query {query!r}: {exc}")
        raise RuntimeError(f"search_papers request failed for query {query!r}: {exc}") from exc
    except ValueError as exc:
        print(f"Invalid JSON returned for query {query!r}: {exc}")
        raise RuntimeError(f"search_papers returned invalid JSON for query {query!r}: {exc}") from exc

    return [Paper(**paper) for paper in result.get("data", []) or []]
