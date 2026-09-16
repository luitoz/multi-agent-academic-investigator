"""Integration tests for src/supervisor.py that call the real supervisor graph end-to-end."""
import pytest

import retrieval_agent
from supervisor import SupervisorState, supervisor_graph


class TestSupervisorGraph:
    @pytest.mark.integration
    def test_full_graph_defines_objectives_and_retrieves_papers(
        self, monkeypatch
    ) -> None:
        """Runs the whole compiled graph (define_objectives -> call_retrieval_agent -> END),
        not just a single node, against the real model and the real retrieval sub-agent."""
        # Keep the integration test cheap/fast; production keeps NUM_DIMENSIONS=2.
        monkeypatch.setattr(retrieval_agent, "NUM_DIMENSIONS", 1)

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
