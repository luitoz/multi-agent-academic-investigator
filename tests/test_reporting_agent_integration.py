"""Test reporting agent behavior end-to-end with a real invocation to the LLM."""
from pathlib import Path

import pytest

from reporting_agent import manage_reporting

INSIGHTS = {
    "supporting_evidence": (
        "Some studies suggest that the relationship between generative AI use and academic "
        "performance is not straightforward. For instance, Kwon and Oh (2026) found that "
        "perceived metacognitive strategy use was not significantly related to writing scores, "
        "indicating that AI use may not directly enhance performance unless paired with critical "
        "monitoring and strategic regulation (Kwon and Oh, 2026)."
    ),
    "conflicting_evidence": (
        "However, other research, such as Saragih (2025), highlights the potential of AI-related "
        "data—such as LMS behavioral metrics—to predict academic performance, suggesting that AI "
        "use may be indirectly linked to performance through engagement and learning behaviors "
        "(Saragih, 2025)."
    ),
    "main_conclusions": (
        "The evidence is inconclusive regarding a direct link between generative AI use and "
        "academic performance. While some studies suggest that AI use may not directly improve "
        "performance without metacognitive regulation (Kwon and Oh, 2026), others indicate that "
        "AI-related data can be predictive of academic outcomes when integrated with other "
        "behavioral metrics (Saragih, 2025)."
    ),
    "methodology": (
        "The methodologies differ significantly: Kwon and Oh (2026) used qualitative and "
        "quantitative analysis of student writing and self-reports, while Saragih (2025) employed "
        "machine learning on synthetic LMS behavioral data. These differences in approach may "
        "explain the divergent findings."
    ),
    "limitations": (
        "Both studies have limitations. Kwon and Oh (2026) relied on a small sample size and "
        "self-reported data, which may affect generalizability. Saragih (2025) used a synthetic "
        "dataset, which may not fully reflect real-world LMS data patterns."
    ),
}

PAPERS = [
    {
        "authors": "Kwon, J., Oh, S.",
        "title": "Metacognitive strategy use and academic writing performance",
        "year": 2026,
        "journal": "Journal of Educational Psychology",
        "volume": "118",
        "pages": "245-260",
        "doi": "https://doi.org/10.1234/kwon2026",
    },
    {
        "authors": "Saragih, A.",
        "title": "Predicting academic performance from LMS behavioral data",
        "year": 2025,
        "journal": "Computers & Education",
        "volume": "212",
        "pages": "104-119",
        "doi": "https://doi.org/10.1234/saragih2025",
    },
]


class TestReportingAgentIntegration:
    # @pytest.mark.integration
    def test_briefing_generation(self, tmp_path: Path) -> None:
        result = manage_reporting.invoke({"insights": INSIGHTS, "papers": PAPERS})

        assert result
        assert "briefing" in result
        assert result["briefing"].strip()
        assert "docx_path" in result
        assert Path(result["docx_path"]).is_file()