from evidence_analysis_agent import RELIABLE_DESCRIPTION, UNRELIABLE_DESCRIPTION
"""

The supervisor only sees the sub-agent's public tool (`schedule_event`) - it
has no knowledge of how the sub-agent validates or retries its own work.
"""
import os
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

from evidence_analysis_agent import manage_evidence_analysis
from framework import model
from retrieval_agent import manage_search
from search_tools import Author, Paper


class SupervisorState(TypedDict):
    request: str
    objectives: NotRequired[list[str]]
    insights: NotRequired[str]
    papers: NotRequired[list[Paper]]
    quality_feedback: NotRequired[str]
    retry_count: NotRequired[int]
    dimensions: NotRequired[list[str]]


NUM_OBJECTIVES = 1
# Retrieval is retried at most once if the gathered papers aren't deemed reliable.
MAX_RETRIEVAL_RETRIES = 1

DEFINE_OBJECTIVES_PROMPT = (
    "You are an academic investigator assistant. Given a natural language research "
    "question, define exactly {num_objectives} research objectives, ordered by "
    "importance, needed to conduct the research on the user's research question."
).format(num_objectives=NUM_OBJECTIVES)

ANALYZE_FINDINGS_PROMPT = (
    "You are an evidence synthesis assistant. Given a list of paper titles and "
    "abstracts, extract the key findings from the gathered evidence: main "
    "conclusions, methodology used, and limitations. Summarize the insights in "
    "clear natural language and markdown format."
)

DEBUG_DUMP_INSIGHTS = os.environ.get("DEBUG_DUMP_INSIGHTS", "").lower() in ("1", "true", "yes")


class ResearchObjectives(BaseModel):
    objectives: list[str] = Field(
        description=(
            f"Exactly {NUM_OBJECTIVES} research objectives, ordered descending by importance, "
            "each phrased as a short goal (e.g., 'Measure the extent of generative "
            "AI use among university students')."
        )
    )


def define_objectives(state: SupervisorState, config: RunnableConfig) -> dict:
    """Step 1: ask the LLM to break the research question into N ordered research objectives."""
    print(f"Defining research objectives for request: {state['request']}")
    objectives_model = model.with_structured_output(ResearchObjectives)
    output = objectives_model.invoke(
        [
            {"role": "system", "content": DEFINE_OBJECTIVES_PROMPT},
            {"role": "user", "content": state["request"]},
        ],
        config=config,
    )
    return {"objectives": output.objectives}

"""
Implementation decision:
Supervisor delegates all information retrieval tasks to the retrieval agent, ensuring that the 
process of gathering relevant academic papers is handled efficiently and systematically. It also makes
a clear separation of concerns between agents, allowing maintainability and extensibility.
"""
def call_retrieval_agent(state: SupervisorState, config: RunnableConfig) -> dict:
    print(f"Calling retrieval agent to gather information for {NUM_OBJECTIVES} ordered research objectives")
    """Step 2: hand the research objectives to the retrieval sub-agent as one natural language request."""
    request = "find papers on this research objectives: " + " ".join(
        f"{objective}." for objective in state.get("objectives", [])
    )
    previous_dimensions = state.get("dimensions") or []
    if state.get("retry_count", 0) > 0 and previous_dimensions:
        request += (
            " This is a retry because the previously gathered papers were not reliable enough. "
            "Identify dimensions different from the ones used last time: "
            + "; ".join(previous_dimensions) + "."
        )
    result = manage_search.invoke({"request": request}, config=config)
    return {
        "papers": result["papers"],
        "dimensions": result["dimensions"],
    }


def call_evidence_analysis_agent(state: SupervisorState, config: RunnableConfig) -> dict:
    """Step 3: assess the reliability of the gathered papers via the evidence analysis sub-agent."""
    print("Calling evidence analysis agent to assess quality of gathered papers")
    result = manage_evidence_analysis.invoke({"papers": state.get("papers", [])}, config=config)
    return {"quality_feedback": result["quality_feedback"]}


def _format_authors(authors: list[Author] | None) -> str:
    """Render an author list as a comma-separated string of names, or 'Unknown' if none."""
    if not authors:
        return "Unknown"
    names = [author.name for author in authors if author.name]
    return ", ".join(names) if names else "Unknown"


def analyze_findings(state: SupervisorState, config: RunnableConfig) -> dict:
    """Step 4: once the gathered papers are deemed reliable, synthesize their findings into insights."""
    papers = state.get("papers") or []
    if not papers:
        raise RuntimeError("No papers were available to analyze.")

    papers_text = "\n\n".join(
        f"Title: {paper.title}\nAuthors: {_format_authors(paper.authors)}\nYear: {paper.year}\n"
        f"Abstract: {paper.abstract}"
        for paper in papers
    )
    print(f"Analyzing {len(papers)} reliable papers and identifying insights")
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

    return {"insights": response.content}


def retry_retrieval(state: SupervisorState) -> dict:
    """Bump the retry counter before re-invoking the retrieval agent."""
    retry_count = state.get("retry_count", 0) + 1
    print(f"Gathered papers were not reliable; retrying retrieval (attempt {retry_count})")
    return {"retry_count": retry_count}


def should_retry_retrieval(state: SupervisorState) -> str:
    """Analyze findings once reliable; otherwise retry retrieval once before giving up."""
    if state.get("quality_feedback") == RELIABLE_DESCRIPTION:
        return "analyze_findings"
    if state.get("retry_count", 0) >= MAX_RETRIEVAL_RETRIES:
        return END
    return "retry_retrieval"


_graph = StateGraph(SupervisorState)
_graph.add_node("define_objectives", define_objectives)
_graph.add_node("call_retrieval_agent", call_retrieval_agent)
_graph.add_node("call_evidence_analysis_agent", call_evidence_analysis_agent)
_graph.add_node("analyze_findings", analyze_findings)
_graph.add_node("retry_retrieval", retry_retrieval)

_graph.add_edge(START, "define_objectives")
_graph.add_edge("define_objectives", "call_retrieval_agent")
_graph.add_edge("call_retrieval_agent", "call_evidence_analysis_agent")
_graph.add_conditional_edges(
    "call_evidence_analysis_agent", should_retry_retrieval, ["analyze_findings", "retry_retrieval", END]
)
_graph.add_edge("analyze_findings", END)
_graph.add_edge("retry_retrieval", "call_retrieval_agent")
supervisor_graph = _graph.compile()

