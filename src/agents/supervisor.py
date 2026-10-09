from evidence_analysis_agent import RELIABLE_DESCRIPTION, UNRELIABLE_DESCRIPTION
"""

The supervisor only sees the sub-agent's public tool (`schedule_event`) - it
has no knowledge of how the sub-agent validates or retries its own work.
"""
import json
import os
from pathlib import Path
from typing import cast

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

from evidence_analysis_agent import manage_evidence_analysis
from framework import logger, model
from reporting_agent import manage_reporting
from retrieval_agent import manage_search
from search_api import Author, Paper

_ROOT_DIR = Path(__file__).resolve().parents[2]
TARGET_DIR = _ROOT_DIR / "target"
TARGET_DIR.mkdir(parents=True, exist_ok=True)


class SupervisorState(TypedDict):
    request: str
    objectives: NotRequired[list[str]]
    insights: NotRequired[dict]
    papers: NotRequired[list[Paper]]
    quality_feedback: NotRequired[str]
    retry_count: NotRequired[int]
    dimensions: NotRequired[list[str]]
    briefing: NotRequired[str]
    report_path: NotRequired[str]


# Default can be overridden via the NUM_RESEARCH_OBJECTIVES environment variable.
NUM_RESEARCH_OBJECTIVES = int(os.environ.get("NUM_RESEARCH_OBJECTIVES", 1))
# Retrieval is retried at most once if the gathered papers aren't deemed reliable.
MAX_RETRIEVAL_RETRIES = int(os.environ.get("MAX_RETRIEVAL_RETRIES", 1))

DEFINE_OBJECTIVES_PROMPT = (
    "You are an academic investigator assistant. Given a natural language research "
    "question, define exactly {num_objectives} research objectives, ordered by "
    "importance, needed to conduct the research on the user's research question."
).format(num_objectives=NUM_RESEARCH_OBJECTIVES)

SYNTHESIZE_FINDINGS_PROMPT = (
"You are an evidence synthesis assistant. Given a set of retrieved academic papers, "
"synthesize the evidence relevant to the research question: {research_question}. Identify supporting and "
"conflicting evidence, overall conclusions, methodologies, and key limitations. "
"Compare findings across studies, highlighting areas of agreement, disagreement, "
"recurring patterns, and methodological differences rather than summarizing papers "
"individually. Support every substantive claim with citations to the retrieved papers "
"using Harvard-style in-text referencing (e.g., Smith, 2024 or Smith and Jones, 2024)."
" Base the synthesis exclusively on the provided evidence and explicitly identify "
"areas where the evidence is insufficient, inconsistent, or inconclusive. "
"Be concise: keep each section to 2-3 sentences."
)



class ResearchObjectives(BaseModel):
    objectives: list[str] = Field(
        description=(
            f"Exactly {NUM_RESEARCH_OBJECTIVES} research objectives, ordered descending by importance, "
            "each phrased as a short goal (e.g., 'Measure the extent of generative "
            "AI use among university students')."
        )
    )


class ResearchFindings(BaseModel):
    supporting_evidence: str = Field(
        description="Evidence across the papers that supports the research question, with Harvard-style citations. "
        "2-3 sentences."
    )
    conflicting_evidence: str = Field(
        description="Evidence across the papers that conflicts or disagrees, with Harvard-style citations. "
        "2-3 sentences."
    )
    main_conclusions: str = Field(
        description="The overall conclusions drawn from synthesizing the evidence. 2-3 sentences."
    )
    methodology: str = Field(
        description="The methodologies used across the papers, compared where relevant. 2-3 sentences."
    )
    limitations: str = Field(
        description="Key limitations of the evidence, including gaps, inconsistencies, or inconclusive areas. "
        "2-3 sentences."
    )


def define_objectives(state: SupervisorState, config: RunnableConfig) -> dict:
    """Step 1: ask the LLM to break the research question into N ordered research objectives."""
    logger.info(f"Defining research objectives for request: {state['request']}")
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
    logger.info(f"Calling retrieval agent to gather information for {NUM_RESEARCH_OBJECTIVES} ordered research objectives")
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
    logger.info("Calling evidence analysis agent to assess quality of gathered papers")
    result = manage_evidence_analysis.invoke({"papers": state.get("papers", [])}, config=config)
    return {"quality_feedback": result["quality_feedback"]}


def _format_authors(authors: list[Author] | None) -> str:
    """Render an author list as a comma-separated string of names, or 'Unknown' if none."""
    if not authors:
        return "Unknown"
    names = [author.name for author in authors if author.name]
    return ", ".join(names) if names else "Unknown"


def synthesize_findings(state: SupervisorState, config: RunnableConfig) -> dict:
    """Step 4: once the gathered papers are deemed reliable, synthesize their findings into insights."""
    papers = state.get("papers") or []
    if not papers:
        raise RuntimeError("No papers were available to analyze.")

    papers_text = "\n\n".join(
        f"Title: {paper.title}\nAuthors: {_format_authors(paper.authors)}\nYear: {paper.year}\n"
        f"Abstract: {paper.abstract}"
        for paper in papers
    )
    logger.info(f"Analyzing {len(papers)} reliable papers and identifying insights")
    findings_model = model.with_structured_output(ResearchFindings)
    output = findings_model.invoke(
        [
            {"role": "system", "content": SYNTHESIZE_FINDINGS_PROMPT.format(research_question=state["request"])},
            {"role": "user", "content": papers_text},
        ],
        config=config,
    )
    insights = cast(ResearchFindings, output).model_dump()
    logger.info(f"Completed analysis of retrieved papers with insights: {insights}")

    return {"insights": insights}


def _papers_for_reporting(papers: list[Paper]) -> list[dict]:
    """Reduce papers to the bibliographic fields the reporting agent needs to build references."""
    return [
        {
            "authors": _format_authors(paper.authors),
            "title": paper.title,
            "year": paper.year,
            "journal": paper.journal.name if paper.journal else None,
            "volume": paper.journal.volume if paper.journal else None,
            "pages": paper.journal.pages if paper.journal else None,
            "doi": paper.externalIds.DOI if paper.externalIds else None,
        }
        for paper in papers
    ]


def call_reporting_agent(state: SupervisorState, config: RunnableConfig) -> dict:
    """Step 5: hand the synthesized insights to the reporting sub-agent to write a natural language briefing to disk."""
    logger.info("Calling reporting agent to write a research briefing from the synthesized insights")
    result = manage_reporting.invoke(
        {
            "insights": state.get("insights", {}),
            "papers": _papers_for_reporting(state.get("papers") or []),
        },
        config=config,
    )
    return {"briefing": result["briefing"], "report_path": result["docx_path"]}


def retry_retrieval(state: SupervisorState) -> dict:
    """Bump the retry counter before re-invoking the retrieval agent."""
    retry_count = state.get("retry_count", 0) + 1
    logger.info(f"Gathered papers were not reliable; retrying retrieval (attempt {retry_count})")
    return {"retry_count": retry_count}


def should_retry_retrieval(state: SupervisorState) -> str:
    """Analyze findings once reliable; otherwise retry retrieval once before giving up."""
    if state.get("quality_feedback") == RELIABLE_DESCRIPTION:
        return "synthesize_findings"
    if state.get("retry_count", 0) >= MAX_RETRIEVAL_RETRIES:
        return END
    return "retry_retrieval"


_graph = StateGraph(SupervisorState)
_graph.add_node("define_objectives", define_objectives)
_graph.add_node("call_retrieval_agent", call_retrieval_agent)
_graph.add_node("call_evidence_analysis_agent", call_evidence_analysis_agent)
_graph.add_node("synthesize_findings", synthesize_findings)
_graph.add_node("call_reporting_agent", call_reporting_agent)
_graph.add_node("retry_retrieval", retry_retrieval)

_graph.add_edge(START, "define_objectives")
_graph.add_edge("define_objectives", "call_retrieval_agent")
_graph.add_edge("call_retrieval_agent", "call_evidence_analysis_agent")
_graph.add_conditional_edges(
    "call_evidence_analysis_agent", should_retry_retrieval, ["synthesize_findings", "retry_retrieval", END]
)
_graph.add_edge("synthesize_findings", "call_reporting_agent")
_graph.add_edge("call_reporting_agent", END)
_graph.add_edge("retry_retrieval", "call_retrieval_agent")
supervisor_graph = _graph.compile()

