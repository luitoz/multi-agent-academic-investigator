"""Evidence analysis sub-agent and the `manage_evidence_analysis` tool that exposes it to the supervisor."""
from datetime import date

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from framework import extract_tool_result, model
from search_tools import Paper

EVALUATE_PAPERS_QUALITY_PROMPT = (
    "You are an evidence quality investigator. Given a list of papers, you should assess whether "
    "those are reliable sources that can support rigorous research. Call assess_evidence_quality "
    "for each paper, provide a quality assessment in clear natural language."
)

# Quality thresholds for assess_evidence_quality; tests may monkeypatch these attributes.
MAX_PUBLICATION_AGE_YEARS = 3
MIN_REFERENCE_COUNT = 10
MIN_CITATION_COUNT = 1


def _has_quality_metrics(paper: Paper) -> bool:
    """Publication must be recent, with enough references and citations."""
    if paper.year is None or paper.referenceCount is None or paper.citationCount is None:
        return False
    is_recent = paper.year >= date.today().year - MAX_PUBLICATION_AGE_YEARS
    result = (
        is_recent
        and paper.referenceCount > MIN_REFERENCE_COUNT
        and paper.citationCount > MIN_CITATION_COUNT
    )
    print(f"Quality metrics for paper '{paper.title}': {result}")
    return result


def _has_complete_journal_info(paper: Paper) -> bool:
    """Journal must report a name, volume, and pages."""
    journal = paper.journal
    result = bool(journal and journal.name and journal.volume and journal.pages)
    print(f"Complete journal info for paper '{paper.title}': {result}")
    return result


def _has_peer_reviewed_type(paper: Paper) -> bool:
    """Publication types must include a journal article or conference paper."""
    publication_types = paper.publicationTypes or []
    result = any(publication_type.lower() in ("journalarticle", "conference") for publication_type in publication_types)
    print(f"Peer-reviewed type for paper '{paper.title}': {result}")
    return result


@tool
def assess_evidence_quality(paper: Paper) -> str:
    """Assess the reliability of a single paper given its bibliographic metadata."""
    checks = {
        "recent publication with enough references and citations": _has_quality_metrics(paper),
        "complete journal name, volume, and pages": _has_complete_journal_info(paper),
        "journal or conference publication type": _has_peer_reviewed_type(paper),
    }
    score = sum(checks.values())
    verdict = "reliable" if score == len(checks) else "questionable" if score > 0 else "unreliable"
    failed = [rule for rule, passed in checks.items() if not passed]
    details = "all checks passed" if not failed else f"failed: {', '.join(failed)}"
    print(f"Assessing evidence quality for paper '{paper.title}': {verdict} ({score}/{len(checks)}) - {details}")
    return verdict


evidence_quality_agent = create_agent(
    model,
    tools=[assess_evidence_quality],
    system_prompt=EVALUATE_PAPERS_QUALITY_PROMPT,
)


class EvidenceAnalysisState(TypedDict):
    papers: list[Paper]
    quality_feedback: str
    messages: list


def assess_evidence_quality_node(state: EvidenceAnalysisState, config: RunnableConfig) -> dict:
    """Let the evidence quality agent assess the gathered papers via the assess_evidence_quality tool."""
    papers = state.get("papers") or []
    if not papers:
        raise RuntimeError("No papers were provided to assess evidence quality.")

    # pass the raw paper JSON so the LLM can call the tool with matching arguments
    papers_text = "\n\n".join(paper.model_dump_json() for paper in papers)
    print(f"Assessing evidence quality for {len(papers)} papers")
    result = evidence_quality_agent.invoke(
        {"messages": [{"role": "user", "content": papers_text}]},
        config=config,
    )
    messages = result["messages"]
    quality_feedback = extract_tool_result(messages, "assess_evidence_quality")
    if not quality_feedback:
        raise RuntimeError("assess_evidence_quality tool was never called.")
    print(f"Evidence quality assessment completed with feedback: {quality_feedback}")
    return {"quality_feedback": quality_feedback, "messages": messages}


def build_evidence_analysis_graph() -> StateGraph:
    graph = StateGraph(EvidenceAnalysisState)
    graph.add_node("assess_evidence_quality", assess_evidence_quality_node)

    graph.add_edge(START, "assess_evidence_quality")
    graph.add_edge("assess_evidence_quality", END)

    return graph

evidence_analysis_graph = build_evidence_analysis_graph().compile()


# entry point for the supervisor to call sub-agent as a tool
@tool
def manage_evidence_analysis(papers: list[Paper], config: RunnableConfig) -> dict:
    """Manage academic evidence quality analysis using previously gathered papers.

    Use this when the user wants to assess the reliability and quality of
    academic papers that have already been retrieved.

    Input: the list of papers gathered by the retrieval agent (e.g., from supervisor state).
    """
    result = evidence_analysis_graph.invoke({"papers": papers}, config=config)
    return {
        "quality_feedback": result["quality_feedback"]
    }

