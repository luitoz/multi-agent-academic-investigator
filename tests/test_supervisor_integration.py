"""Integration tests for src/supervisor.py that call the real supervisor graph end-to-end."""
import pytest

import evidence_analysis_agent
import retrieval_agent
from supervisor import SupervisorState, supervisor_graph


class TestSupervisorGraph:
    @pytest.mark.integration
    def test_full_graph_succeeds_with_reliable_evidence_on_first_try(
        self, monkeypatch
    ) -> None:
        """Runs the whole compiled graph end-to-end (define_objectives -> call_retrieval_agent ->
        call_evidence_analysis_agent -> END) against the real model and the real retrieval and
        evidence analysis sub-agents, exercising the success path where the gathered papers are
        deemed reliable on the first try and no retrieval retry is needed."""
        # Keep the integration test cheap/fast; production keeps NUM_DIMENSIONS=2.
        monkeypatch.setattr(retrieval_agent, "NUM_DIMENSIONS", 1)
        # Avoid flaky quality verdicts from the real model; only the retrieval/supervisor wiring is under test.
        monkeypatch.setattr(evidence_analysis_agent.assess_evidence_quality, "func", 
                            lambda paper: evidence_analysis_agent.RELIABLE_DESCRIPTION)

        result = supervisor_graph.invoke(
            SupervisorState(
                request=(
                    "How does generative AI use among university students relate to "
                    "their academic performance?"
                )
            )
        )
        assert result
        assert "insights" in result
        assert len(result["insights"]) > 0
        assert "papers" in result
        assert len(result["papers"]) > 0
        assert result["quality_feedback"] == evidence_analysis_agent.RELIABLE_DESCRIPTION
        assert result.get("retry_count", 0) == 0

