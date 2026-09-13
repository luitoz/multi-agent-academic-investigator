""" Test retrieval agent behavior end-to-end with real invocations to the LLM and search tools.

"""
import pytest

from retrieval_agent import manage_search


class TestManageSearch:
    # @pytest.mark.integration
    def test_search_covers_multiple_dimensions(self) -> None:
        """Runs the real retrieval agent end-to-end against a natural language request."""
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
        assert len(result["dimensions"]) >= 3
        assert len(result["queries"]) == len(result["dimensions"])
        assert len(result["tool_results"]) == len(result["queries"])
