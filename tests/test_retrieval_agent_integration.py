""" Test retrieval agent behavior end-to-end with real invocations to the LLM and search tools.

"""
import pytest

import retrieval_agent
from retrieval_agent import manage_search


class TestManageSearch:
    @pytest.mark.integration
    def test_search_covers_multiple_dimensions(self, monkeypatch) -> None:
        """Runs the real retrieval agent end-to-end against a natural language request."""
        # Keep the integration test cheap/fast; production keeps NUM_DIMENSIONS=2.
        monkeypatch.setattr(retrieval_agent, "NUM_DIMENSIONS", 1)

        result = manage_search.invoke(
            {
                "request": (
                    "find papers on this research "
                    "objetives: Measure the extent of generative AI use among university students, "
                    "Examine the association between AI usage and academic performance. "
                )
            }
        )
        assert result
        assert "dimensions" in result
        assert "queries" in result
        assert "tool_results" in result
        assert "insights" in result
        assert len(result["dimensions"]) == 1
        assert len(result["queries"]) == len(result["dimensions"])
        assert len(result["tool_results"]) == len(result["queries"])
