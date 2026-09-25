"""Unit tests for src/retrieval_agent.py.

Test the deterministic code behaviour of the retrieval agent functions mocking
the LLM and search tool calls.
"""
from typing import cast

import pytest

import retrieval_agent as retrieval_agent_module
from retrieval_agent import (
    RetrievalState,
    SearchDimensions,
    _dimension_to_query,
    identify_dimensions,
    manage_search,
    search_dimensions,
)
from search_api import Paper, search_papers




class TestDimensionToQuery:
    def test_joins_words_with_plus(self):
        assert _dimension_to_query("generative AI use among university students") == (
            "generative+ai+use+among+university+students"
        )

    def test_strips_punctuation(self):
        assert _dimension_to_query("COVID-19, vaccination & Europe!") == "covid+19+vaccination+europe"

    def test_excludes_common_stopwords(self):
        assert _dimension_to_query("the use of AI in the classroom") == "use+ai+classroom"

    def test_empty_dimension_returns_empty_string(self):
        assert _dimension_to_query("") == ""


class TestIdentifyDimensions:
    def test_returns_dimensions_from_structured_output(self, monkeypatch):
        expected = SearchDimensions(dimensions=["dim one", "dim two", "dim three"])

        class FakeStructuredModel:
            def invoke(self, _messages, config=None):
                return expected

        monkeypatch.setattr(
            type(retrieval_agent_module.model),
            "with_structured_output",
            lambda self, _schema: FakeStructuredModel(),
        )

        state = cast(RetrievalState, {"request": "find papers on AI in education"}) #initial state for the test
        result = identify_dimensions(state, {})

        assert result == {"dimensions": expected.dimensions}


class TestSearchDimensions:
    def test_builds_one_query_and_result_per_dimension(self, monkeypatch):
        def fake_search(query):
            return [Paper(title=f"Title for {query}", abstract=f"Abstract for {query}")]

        monkeypatch.setattr(search_papers, "func", fake_search)

        state = cast(RetrievalState, {"dimensions": ["AI in education", "student performance"]})
        result = search_dimensions(state)

        assert result["queries"] == ["ai+education", "student+performance"]
        assert result["tool_results"] == [
            fake_search("ai+education"),
            fake_search("student+performance"),
        ]
        assert [paper.title for paper in result["papers"]] == [
            "Title for ai+education",
            "Title for student+performance",
        ]

    def test_raises_when_no_papers_have_title_and_abstract(self, monkeypatch):
        monkeypatch.setattr(search_papers, "func", lambda query: [Paper(title=None, abstract=None)])

        state = cast(RetrievalState, {"dimensions": ["AI in education"]})

        with pytest.raises(RuntimeError, match="No paper titles or abstracts were available"):
            search_dimensions(state)


