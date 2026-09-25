"""Runs the whole compiled graph (assess_evidence_quality -> END) against the real model,
letting it decide when to call the assess_evidence_quality tool."""
import pytest

from evidence_analysis_agent import QUESTIONABLE_DESCRIPTION, RELIABLE_DESCRIPTION, UNRELIABLE_DESCRIPTION

from evidence_analysis_agent import manage_evidence_analysis
from search_api import Journal, Paper
@pytest.mark.integration

class TestManageEvidenceAnalysis:
    def test_assesses_quality_of_gathered_papers_as_reliable(self) -> None:
        """Runs the whole compiled graph (assess_evidence_quality -> END) against the real model,
        letting it decide when to call the assess_evidence_quality tool."""
        papers = [
            Paper(
                paperId="paper-1",
                title="Generative AI use among university students",
                abstract=(
                    "This paper studies the extent of generative AI use among university "
                    "students and its association with academic performance."
                ),
                year=2023,
                referenceCount=42,
                citationCount=17,
                publicationTypes=["JournalArticle"],
                journal=Journal(name="Journal of Educational Technology", volume="12", pages="1-20"),
                venue="Journal of Educational Technology",
            ),
            Paper(
                paperId="paper-3",
                title="Impact of large language models on peer review",
                abstract=(
                    "This paper examines how large language models are reshaping the academic "
                    "peer review process across several disciplines."
                ),
                year=2024,
                referenceCount=35,
                citationCount=12,
                publicationTypes=["Conference"],
                journal=Journal(name="Conference on AI in Academia", volume="5", pages="100-115"),
                venue="Conference on AI in Academia",
            ),
        ]

        result = manage_evidence_analysis.invoke({"papers": papers})

        assert result
        assert "quality_feedback" in result
        assert result["quality_feedback"] == RELIABLE_DESCRIPTION

    def test_assesses_quality_of_gathered_papers_as_unreliable(self) -> None:
        """A paper that is old, poorly cited, lacks journal info, and isn't a journal/conference
        publication should fail every quality check and be reported as unreliable."""
        papers = [
            Paper(
                paperId="paper-2",
                title="An obscure preprint on AI in education",
                abstract="This preprint speculates about AI use in education without empirical data.",
                year=2005,
                referenceCount=2,
                citationCount=0,
                publicationTypes=["Review"],
                journal=None,
                venue="",
            )
        ]

        result = manage_evidence_analysis.invoke({"papers": papers})

        assert result
        assert "quality_feedback" in result
        assert result["quality_feedback"] == UNRELIABLE_DESCRIPTION

    def test_assesses_quality_of_gathered_papers_as_questionable(self) -> None:
        """A mix of a reliable paper and an unreliable paper should yield an overall
        questionable verdict, since not every paper is reliable nor is every paper unreliable."""
        papers = [
            Paper(
                paperId="paper-1",
                title="Generative AI use among university students",
                abstract=(
                    "This paper studies the extent of generative AI use among university "
                    "students and its association with academic performance."
                ),
                year=2023,
                referenceCount=42,
                citationCount=17,
                publicationTypes=["JournalArticle"],
                journal=Journal(name="Journal of Educational Technology", volume="12", pages="1-20"),
                venue="Journal of Educational Technology",
            ),
            Paper(
                paperId="paper-2",
                title="An obscure preprint on AI in education",
                abstract="This preprint speculates about AI use in education without empirical data.",
                year=2005,
                referenceCount=2,
                citationCount=0,
                publicationTypes=["Review"],
                journal=None,
                venue="",
            ),
        ]

        result = manage_evidence_analysis.invoke({"papers": papers})

        assert result
        assert "quality_feedback" in result
        assert result["quality_feedback"] == QUESTIONABLE_DESCRIPTION

