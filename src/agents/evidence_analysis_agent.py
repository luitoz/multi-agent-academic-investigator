"""Evidence analysis sub-agent and the `manage_evidence_analysis` tool that exposes it to the supervisor."""
from datetime import date
import json
import os

from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from framework import logger, shallow_paper_json
from search_api import Paper

# Quality thresholds for assess_evidence_quality; tests may monkeypatch these attributes.
# Defaults can be overridden via environment variables.
MAX_PUBLICATION_AGE_YEARS = int(os.environ.get("MAX_PUBLICATION_AGE_YEARS", 3))
MIN_REFERENCE_COUNT = int(os.environ.get("MIN_REFERENCE_COUNT", 1))
MIN_CITATION_COUNT = int(os.environ.get("MIN_CITATION_COUNT", 1))
RELIABLE_DESCRIPTION = 'reliable'
UNRELIABLE_DESCRIPTION = 'unreliable'
QUESTIONABLE_DESCRIPTION = 'questionable'

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
    logger.info(f"Quality metrics for paper '{json.dumps(shallow_paper_json(paper), indent=2)}': {result}")
    return result


def _has_complete_journal_info(paper: Paper) -> bool:
    """Journal must report a name, volume, and pages, and the paper must have a DOI."""
    logger.info(f"Journal info for paper '{paper.title}': {paper.journal}")
    journal = paper.journal
    has_doi = bool(paper.externalIds and paper.externalIds.DOI)
    result = bool(journal and journal.name and journal.volume and journal.pages and has_doi)
    logger.info(f"Complete journal info for paper '{paper.title}': {result}")
    return result


def _has_peer_reviewed_type(paper: Paper) -> bool:
    """Publication types must include a journal article or conference paper."""
    logger.info(f"Publication types for paper '{paper.title}': {paper.publicationTypes}")
    publication_types = paper.publicationTypes or []
    result = any(publication_type.lower() in ("journalarticle", "conference") for publication_type in publication_types)
    logger.info(f"Peer-reviewed type for paper '{paper.title}': {result}")
    return result


@tool
def assess_evidence_quality(paper: Paper) -> str:
    """Assess the reliability of a single paper given its bibliographic metadata."""
    checks = {
        "recent publication with enough references and citations": _has_quality_metrics(paper),
        "journal or conference publication type": _has_peer_reviewed_type(paper),
        "complete journal name, volume, pages, and DOI": _has_complete_journal_info(paper),
    }
    score = sum(checks.values())
    verdict = RELIABLE_DESCRIPTION if score == len(checks) else QUESTIONABLE_DESCRIPTION if score > 0 else UNRELIABLE_DESCRIPTION
    failed = [rule for rule, passed in checks.items() if not passed]
    details = "all checks passed" if not failed else f"failed: {', '.join(failed)}"
    logger.info(f"Assessing evidence quality for paper '{paper.title}': {verdict} ({score}/{len(checks)}) - {details}")
    return verdict


class EvidenceAnalysisState(TypedDict):
    papers: list[Paper]
    quality_feedback: str

""" implementation decision: make evidence quality assessment fully deterministic to
  economize token usage and reduce response time. """
def assess_evidence_quality_node(state: EvidenceAnalysisState, config: RunnableConfig) -> dict:
    """Call the assess_evidence_quality tool directly for each paper, without going through an LLM."""
    papers = state.get("papers") or []
    if not papers:
        raise RuntimeError("No papers were provided to assess evidence quality.")

    logger.info(f"Assessing evidence quality for {len(papers)} papers")
    tool_results = [assess_evidence_quality.invoke({"paper": paper}, config=config) for paper in papers]
    if all(r == RELIABLE_DESCRIPTION for r in tool_results):
        quality_feedback = RELIABLE_DESCRIPTION
    elif all(r == UNRELIABLE_DESCRIPTION for r in tool_results):
        quality_feedback = UNRELIABLE_DESCRIPTION
    else:
        quality_feedback = QUESTIONABLE_DESCRIPTION
    logger.info(f"Evidence quality assessment for {len(papers)} papers completed with feedback: {quality_feedback}")
    return {"quality_feedback": quality_feedback}


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

