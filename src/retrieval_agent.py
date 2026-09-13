"""Retrieval sub-agent and the `manage_search` tool that exposes it to the supervisor."""
import re

from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from framework import model


@tool
def search_papers(
    query: str
) -> str:
    """Search for academic papers using the given query."""
    # Stub: In practice, this would call an academic database API.
    return f"Search results for query: {query}"


IDENTIFY_DIMENSIONS_PROMPT = (
    "You are an evidence retrieval assistant. Given a natural language research "
    "request, identify at least 3 distinct dimensions that should be covered to "
    "comprehensively accomplish the research objectives."
)


class SearchDimensions(BaseModel):
    dimensions: list[str] = Field(
        description=(
            "At least 3 distinct dimensions covering the research objectives, "
            "each phrased as a short topic (e.g., 'generative AI use among university students')."
        )
    )


class RetrievalState(TypedDict):
    request: str
    dimensions: list[str]
    queries: list[str]
    tool_results: list[str]
    result: str


def identify_dimensions(state: RetrievalState, config: RunnableConfig) -> dict:
    """Step 1: ask the LLM to break the request into distinct search dimensions."""
    dimensions_model = model.with_structured_output(SearchDimensions)
    output = dimensions_model.invoke(
        [
            {"role": "system", "content": IDENTIFY_DIMENSIONS_PROMPT},
            {"role": "user", "content": state["request"]},
        ],
        config=config,
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
    return {"queries": queries, "tool_results": tool_results, "result": "\n".join(tool_results)}


def build_retrieval_graph() -> StateGraph:
    graph = StateGraph(RetrievalState)
    graph.add_node("identify_dimensions", identify_dimensions)
    graph.add_node("search_dimensions", search_dimensions)

    graph.add_edge(START, "identify_dimensions")
    graph.add_edge("identify_dimensions", "search_dimensions")
    graph.add_edge("search_dimensions", END)

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
        "result": result["result"],
        "dimensions": result["dimensions"],
        "queries": result["queries"],
        "tool_results": result["tool_results"],
    }
    
