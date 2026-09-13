"""Retrieval sub-agent and the `manage_search` tool that exposes it to the supervisor."""
import os
import re
from pathlib import Path
from typing import Protocol, cast

from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from framework import model
import requests
import json

# Default dimension count for real execution; tests may monkeypatch this attribute.
NUM_DIMENSIONS = 2


def _identify_dimensions_prompt(num_dimensions: int) -> str:
    return (
        "You are an evidence retrieval assistant. Given a natural language research "
        f"request, identify exactly {num_dimensions} distinct dimensions that should be covered to "
        "comprehensively accomplish the research objectives."
    )


ANALYZE_FINDINGS_PROMPT = (
    "You are an evidence synthesis assistant. Given a list of paper titles and "
    "abstracts, extract the key findings from the gathered evidence: main "
    "conclusions, methodology used, and limitations. Summarize the insights in "
    "clear natural language."
)

DEBUG_DUMP_INSIGHTS = os.environ.get("DEBUG_DUMP_INSIGHTS", "").lower() in ("1", "true", "yes")

@tool
def search_papers(
    query: str
) -> dict:
    """Search for academic papers using the given query."""

    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    query_params = {
        "query": query,
        "limit": 1,
        "fields": "paperId,title,abstract,year,referenceCount,citationCount,isOpenAccess,fieldsOfStudy"
    }
    # TODO ask for a the key when running production. for integration tests, use a mock service
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    print(f"Using Semantic Scholar API key: {api_key}")

    # Define headers with API key
    headers = {"x-api-key": api_key}

    try:
        response = requests.get(url, params=query_params, headers=headers, timeout=10)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"search_papers request failed for query {query!r}: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(f"search_papers returned invalid JSON for query {query!r}: {exc}") from exc

    print(result)
    return result




class SearchDimensions(BaseModel):
    dimensions: list[str] = Field(
        description=(
            f"Exactly {NUM_DIMENSIONS} distinct dimensions covering the research objectives, "
            "each phrased as a short topic (e.g., 'generative AI use among university students')."
        )
    )


def _build_search_dimensions_schema(num_dimensions: int) -> type[BaseModel]:
    """Build a schema whose description reflects `num_dimensions` (kept in sync when NUM_DIMENSIONS is overridden)."""
    class _SearchDimensions(BaseModel):
        dimensions: list[str] = Field(
            description=(
                f"Exactly {num_dimensions} distinct dimensions covering the research objectives, "
                "each phrased as a short topic (e.g., 'generative AI use among university students')."
            )
        )

    return _SearchDimensions


class _DimensionsResult(Protocol):
    dimensions: list[str]


class RetrievalState(TypedDict):
    request: str
    dimensions: list[str]
    queries: list[str]
    tool_results: list[dict]
    insights: str


def identify_dimensions(state: RetrievalState, config: RunnableConfig) -> dict:
    """Step 1: ask the LLM to break the request into distinct search dimensions."""
    num_dimensions = NUM_DIMENSIONS
    schema = _build_search_dimensions_schema(num_dimensions)
    dimensions_model = model.with_structured_output(schema)
    output = cast(
        _DimensionsResult,
        dimensions_model.invoke(
            [
                {"role": "system", "content": _identify_dimensions_prompt(num_dimensions)},
                {"role": "user", "content": state["request"]},
            ],
            config=config,
        ),
    )
    return {"dimensions": output.dimensions}


def _dimension_to_query(dimension: str) -> str:
    """Turn a dimension description into a `+`-joined search query, e.g. 'covid+vaccination+europe'."""
    words = re.findall(r"[a-zA-Z0-9]+", dimension.lower())
    return "+".join(words)


def search_dimensions(state: RetrievalState) -> dict:
    """Step 2: build one query per dimension and invoke search_papers once per query."""
    queries = [_dimension_to_query(dimension) for dimension in state["dimensions"]]
    tool_results = [search_papers.invoke({"query": query}) for query in queries]

    return {"queries": queries, "tool_results": tool_results}



def _extract_papers(tool_results: list[dict]) -> list[dict]:
    """Pull out just the title and abstract fields from each search_papers response."""
    papers = []
    for result in tool_results:
        for paper in (result or {}).get("data", []) or []:
            title = paper.get("title")
            abstract = paper.get("abstract")
            if title and abstract:
                papers.append({"title": title, "abstract": abstract})
    return papers


def analyze_findings(state: RetrievalState, config: RunnableConfig) -> dict:
    """Step 3: analyze the retrieved papers' titles/abstracts to surface conclusions, methodology, and limitations."""
    papers = _extract_papers(state["tool_results"])
    if not papers:
        return {"insights": "No paper titles or abstracts were available to analyze."}

    papers_text = "\n\n".join(
        f"Title: {paper['title']}\nAbstract: {paper['abstract']}" for paper in papers
    )
    response = model.invoke(
        [
            {"role": "system", "content": ANALYZE_FINDINGS_PROMPT},
            {"role": "user", "content": papers_text},
        ],
        config=config,
    )

    # dump raw LLM output for debugging formatting/content issues
    # TODO ask explicitly for markdown formatted output
    if DEBUG_DUMP_INSIGHTS:
        debug_dir = Path(__file__).resolve().parent.parent / "debug"
        debug_dir.mkdir(exist_ok=True)
        (debug_dir / "insights.md").write_text(response.content)

    return {"insights": response.content}


def build_retrieval_graph() -> StateGraph:
    graph = StateGraph(RetrievalState)
    graph.add_node("identify_dimensions", identify_dimensions)
    graph.add_node("search_dimensions", search_dimensions)
    graph.add_node("analyze_findings", analyze_findings)

    graph.add_edge(START, "identify_dimensions")
    graph.add_edge("identify_dimensions", "search_dimensions")
    graph.add_edge("search_dimensions", "analyze_findings")
    graph.add_edge("analyze_findings", END)

    return graph

retrieval_graph = build_retrieval_graph().compile()


# entry point for the supervisor to call sub-agent as a tool
@tool
def manage_search(request: str, config: RunnableConfig) -> dict:
    """Manage academic search requests using natural language.

    Use this when the user wants to search for academic papers.
    Handles search dimension identification, query generation and submission to 
    the academic database retrieval tool.

    Input: Natural language search request (e.g., 'find papers on this research 
        objetives: Measure the extent of generative AI use among university students, 
        Examine the association between AI usage and academic performance. ')
    """
    result = retrieval_graph.invoke({"request": request}, config=config)
    return {
        "dimensions": result["dimensions"],
        "queries": result["queries"],
        "tool_results": result["tool_results"],
        "insights": result["insights"],
    }
    
