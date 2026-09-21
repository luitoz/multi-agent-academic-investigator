""" Test retrieval agent behavior end-to-end with real invocations to the LLM and search tools.

"""
import pytest

import retrieval_agent
from retrieval_agent import manage_search

@pytest.mark.integration
class TestManageSearch:
    
    def test_search_covers_multiple_dimensions_using_mock_endpoint_server_with_success_response(
            self, monkeypatch, mock_semantic_scholar_server) -> None:
        """Runs the retrieval agent end-to-end but using the mock Semantic Scholar endpoint server to
        economize on API calls."""

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
        assert len(result["queries"]) == len(result["dimensions"])
        assert len(result["tool_results"]) == len(result["queries"])  
        assert "papers" in result
        assert result["papers"][0].paperId.startswith("mock")
        assert result["papers"][0].journal is not None
        assert result["papers"][0].journal.name == "Mock Journal"
        assert result["papers"][0].authors is not None
        assert result["papers"][0].authors[0].name == "Mock Author"
        assert result["papers"][0].authors[0].authorId == "mock-author-1"

    def test_search_covers_multiple_dimensions_using_mock_endpoint_server_with_error_response(
            self, monkeypatch, mock_semantic_scholar_usage_limit_server) -> None:
        """Runs the retrieval agent end-to-end but using the mock Semantic Scholar endpoint server to
        simulate a usage limit error and economize on API calls."""
        with pytest.raises(RuntimeError, match="search_papers request failed"):
            manage_search.invoke(
                {
                    "request": (
                        "find papers on this research "
                        "objetives: Measure the extent of generative AI use among university students, "
                        "Examine the association between AI usage and academic performance. "
                    )
                }
            )
        