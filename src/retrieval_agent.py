"""Retrieval sub-agent and the `manage_search` tool that exposes it to the supervisor."""
import os
import re
from pathlib import Path
from typing import Protocol, cast

import nltk
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from nltk.corpus import stopwords
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from framework import model
from search_tools import Paper, search_papers

# Default dimension count for real execution; tests may monkeypatch this attribute.
NUM_DIMENSIONS = 2


def _load_stopwords() -> set[str]:
    """Load the NLTK English stopwords, downloading them first if not already present."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        return set(stopwords.words("english"))


_STOPWORDS = _load_stopwords()


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
    "clear natural language and markdown format."
)

DEBUG_DUMP_INSIGHTS = os.environ.get("DEBUG_DUMP_INSIGHTS", "").lower() in ("1", "true", "yes")


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
    tool_results: list[list[Paper]]
    insights: str
    papers: list[Paper]


def identify_dimensions(state: RetrievalState, config: RunnableConfig) -> dict:
    """Step 1: ask the LLM to break the request into distinct search dimensions."""
    print(f"Identifying dimensions for request: {state['request']}")
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
    print(f"Identified {NUM_DIMENSIONS} dimensions:")
    return {"dimensions": output.dimensions}


def _dimension_to_query(dimension: str) -> str:
    """Turn a dimension description into a `+`-joined search query, e.g. 'covid+vaccination+europe'."""
    words = re.findall(r"[a-zA-Z0-9]+", dimension.lower())
    words = [word for word in words if word not in _STOPWORDS]
    return "+".join(words)

"""
Implementation decision: Deterministic code will handle paper retrieval to minimize unnecessary
 requests to rate-limited academic search APIs.
"""
def search_dimensions(state: RetrievalState) -> dict:
    """Step 2: build one query per dimension and invoke search_papers once per query."""
    queries = [_dimension_to_query(dimension) for dimension in state["dimensions"]]
    print("Searching identified dimensions with tool")
    tool_results = [search_papers.invoke({"query": query}) for query in queries]
    print(f"Completed search for {len(queries)} queries")
    return {"queries": queries, "tool_results": tool_results}



def _extract_papers(tool_results: list[list[Paper]]) -> list[Paper]:
    """Flatten the per-query results, keeping only papers that have both a title and abstract."""
    papers = []
    for result in tool_results:
        for paper in result or []:
            if paper.title and paper.abstract:
                papers.append(paper)
    return papers


def analyze_findings(state: RetrievalState, config: RunnableConfig) -> dict:
    """Step 3: analyze the retrieved papers' titles/abstracts to surface conclusions, methodology, and limitations."""
    papers = _extract_papers(state["tool_results"])
    if not papers:
        raise RuntimeError("No paper titles or abstracts were available to analyze. Try another research question.") 

    papers_text = "\n\n".join(
        f"Title: {paper.title}\nAbstract: {paper.abstract}" for paper in papers
    )
    print(f"Analyzing {len(papers)} retrieved papers and identifying insights")
    response = model.invoke(
        [
            {"role": "system", "content": ANALYZE_FINDINGS_PROMPT},
            {"role": "user", "content": papers_text},
        ],
        config=config,
    )
    print("Completed analysis of retrieved papers")
    # dump raw LLM output for debugging formatting/content issues
    if DEBUG_DUMP_INSIGHTS:
        debug_dir = Path(__file__).resolve().parent.parent / "debug"
        debug_dir.mkdir(exist_ok=True)
        (debug_dir / "insights.md").write_text(response.content)

    return {"insights": response.content, "papers": papers}


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
        "papers": result["papers"]
    }
    
