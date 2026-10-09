"""Reporting sub-agent and the `manage_reporting` tool that exposes it to the supervisor."""
from datetime import datetime
from pathlib import Path

import pypandoc
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

from framework import logger, model

# Default location briefings are written to when the caller doesn't supply an output_path.
REPORTS_DIR = Path(__file__).resolve().parents[2] / "target"

BRIEFING_PROMPT = (
    "You are a research communications assistant.\n\n"
    "Given a JSON object containing synthesized research insights, produce a clear, concise, "
    "and well-structured research briefing for a non-technical stakeholder.\n\n"
    "Requirements:\n"
    "- Write in accessible, natural language and minimal technical jargon.\n"
    "- Organize the briefing with short, descriptive section headings.\n"
    "- Include a clear and informative title.\n"
    "- Include a prominently highlighted \"Main Conclusions\" section that summarizes the most "
    "important takeaways from the evidence.\n"
    "- Preserve all in-text citations provided in the input and associate them with the claims "
    "they support.\n"
    "- Include a \"References\" section at the end, with references formatted in Harvard style "
    "using only bibliographic information available in the input, including the journal's "
    "volume, pages and DOI when they are provided. DOI must be formatted as a valid URL.\n"
    "- Do not invent, infer, or add evidence, findings, citations, references, or bibliographic "
    "details that are not present in the input.\n"
    "- Clearly represent uncertainty, limitations, conflicting evidence, or lack of consensus "
    "when these are present in the synthesized insights.\n"
    "- Prioritize the findings most relevant to the research objective rather than mechanically "
    "reproducing the structure of the input JSON.\n\n"
    "- Produce briefing in markdown format.\n"
    "Required structure:\n"
    "1. Begin with a clear, informative title formatted as the document's only level-1 heading (`#`). "
    "All subsequent section headings must be level-2 headings (`##`) or lower.\n"
    "2. Main Conclusions\n"
    "3. Evidence and Key Findings\n"
    "4. Limitations, Uncertainty, or Conflicting Evidence (when applicable)\n"
    "5. References\n\n"
    "The final briefing should be self-contained, coherent, and understandable without requiring "
    "the reader to inspect the source JSON."
)


class ResearchBriefing(BaseModel):
    briefing: str = Field(
        description=(
            "A well-structured, natural language research briefing summarizing the provided "
            "insights for a non-technical stakeholder, preserving any in-text citations."
        )
    )


class ReportingState(TypedDict):
    insights: dict
    papers: NotRequired[list[dict]]
    output_path: NotRequired[str]
    briefing: NotRequired[str]
    docx_path: NotRequired[str]


def generate_briefing(state: ReportingState, config: RunnableConfig) -> dict:
    """Step 1: ask the LLM to turn the structured insights into a natural language briefing."""
    insights = state.get("insights") or {}
    if not insights:
        raise RuntimeError("No insights were provided to generate a research briefing.")

    # Bibliographic data used only to build the References section; kept separate from the
    # synthesized insights so the LLM doesn't mistake it for additional evidence.
    papers = state.get("papers") or []
    user_content = str(insights)
    if papers:
        user_content += "\n\nReferences source data:\n" + str(papers)

    logger.info("Generating natural language research briefing from insights")
    briefing_model = model.with_structured_output(ResearchBriefing)
    output = briefing_model.invoke(
        [
            {"role": "system", "content": BRIEFING_PROMPT},
            {"role": "user", "content": user_content},
        ],
        config=config,
    )
    return {"briefing": output.briefing}


def _default_output_path() -> str:
    """Build a timestamped default path under the repo-level `reports/` directory."""
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    return str(REPORTS_DIR / f"research_briefing_{timestamp}.md")


""" implementation decision: writing to disk is deterministic I/O, kept out of the LLM's hands. """
def write_briefing(state: ReportingState) -> dict:
    """Step 2: write the briefing to markdown, convert it to .docx via pypandoc, then remove the
    markdown source, keeping only the docx path in state.

    Uses the pandoc binary bundled by the pypandoc-binary package, so no system-wide
    pandoc installation is required.
    """
    briefing = state.get("briefing") or ""
    if not briefing:
        raise RuntimeError("No briefing text was generated to write to disk.")

    output_path = Path(state.get("output_path") or _default_output_path())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(briefing, encoding="utf-8")
    logger.info(f"Wrote research briefing to {output_path}")

    docx_path = output_path.with_suffix(".docx")
    try:
        pypandoc.convert_file(
            str(output_path), "docx", format="markdown-auto_identifiers", outputfile=str(docx_path)
        )
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"pandoc failed to convert briefing to docx: {exc}") from exc
    output_path.unlink()
    logger.info(f"Exported research briefing to {docx_path} and removed {output_path}")
    return {"docx_path": str(docx_path)}


def build_reporting_graph() -> StateGraph:
    graph = StateGraph(ReportingState)
    graph.add_node("generate_briefing", generate_briefing)
    graph.add_node("write_briefing", write_briefing)

    graph.add_edge(START, "generate_briefing")
    graph.add_edge("generate_briefing", "write_briefing")
    graph.add_edge("write_briefing", END)

    return graph

reporting_graph = build_reporting_graph().compile()


# entry point for the supervisor to call sub-agent as a tool
@tool
def manage_reporting(
    insights: dict,
    config: RunnableConfig,
    papers: list[dict] | None = None,
    output_path: str | None = None,
) -> dict:
    """Manage generation of a natural language research briefing from synthesized insights.

    Use this when the user wants a human-readable report written to disk from
    previously synthesized research insights.

    Input: the insights dict produced by the evidence synthesis step (e.g., from
        supervisor state), an optional list of paper dicts (authors, title, year, journal,
        volume, pages and DOI) used to build the References section, and optionally a local file path to write the
        briefing to (defaults to a timestamped file under the repo-level `reports/` directory).
        The briefing is exported to a .docx file using the pandoc binary bundled with the
        pypandoc-binary package, and the intermediate markdown file is then removed.
    """
    result = reporting_graph.invoke(
        {"insights": insights, "papers": papers or [], "output_path": output_path}, config=config
    )
    return {
        "briefing": result["briefing"],
        "docx_path": result["docx_path"],
    }
