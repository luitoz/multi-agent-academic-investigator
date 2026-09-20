"""Unit tests for the assess_evidence_quality tool in src/evidence_analysis_agent.py."""
from datetime import date

import evidence_analysis_agent as evidence_analysis_agent_module
from evidence_analysis_agent import assess_evidence_quality
from search_tools import Journal, Paper


def _make_paper(**overrides) -> Paper:
    defaults = dict(
        paperId="paper-1",
        title="Sample paper",
        abstract="Sample abstract.",
        year=date.today().year,
        referenceCount=20,
        citationCount=5,
        publicationTypes=["JournalArticle"],
        journal=Journal(name="Sample Journal", volume="1", pages="1-10"),
        venue="Sample Venue",
    )
    defaults.update(overrides)
    return Paper(**defaults)


class TestAssessEvidenceQuality:
    def test_returns_reliable_when_all_checks_pass(self):
        paper = _make_paper()
        assert assess_evidence_quality.func(paper) == evidence_analysis_agent_module.RELIABLE_DESCRIPTION

    def test_returns_unreliable_when_all_checks_fail(self):
        paper = _make_paper(
            year=2000,
            referenceCount=0,
            citationCount=0,
            publicationTypes=["Review"],
            journal=None,
        )
        assert assess_evidence_quality.func(paper) == evidence_analysis_agent_module.UNRELIABLE_DESCRIPTION

    def test_returns_questionable_when_some_checks_fail(self):
        paper = _make_paper(journal=None)
        assert assess_evidence_quality.func(paper) == evidence_analysis_agent_module.QUESTIONABLE_DESCRIPTION

    def test_conference_publication_type_counts_as_peer_reviewed(self):
        paper = _make_paper(publicationTypes=["Conference"], journal=None)
        assert assess_evidence_quality.func(paper) == evidence_analysis_agent_module.QUESTIONABLE_DESCRIPTION

    def test_respects_monkeypatched_thresholds(self, monkeypatch):
        monkeypatch.setattr(evidence_analysis_agent_module, "MIN_REFERENCE_COUNT", 100)
        paper = _make_paper()
        assert assess_evidence_quality.func(paper) == evidence_analysis_agent_module.QUESTIONABLE_DESCRIPTION
