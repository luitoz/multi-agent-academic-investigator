"""Integration tests for src/supervisor.py that call the real supervisor graph end-to-end."""
import socket
import threading
import time
from http.server import HTTPServer

import pytest
import search_api

import evidence_analysis_agent
import mockserver
import retrieval_agent
from supervisor import SupervisorState, supervisor_graph

MOCK_SERVER_PORT = 8000


def _mock_server_running(host: str = "localhost", port: int = MOCK_SERVER_PORT) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


@pytest.fixture(scope="module", autouse=True)
def ensure_mock_server():
    """Start the local mock Semantic Scholar server if it isn't already running, and stop it afterward
    if this fixture is the one that started it (leaves an externally-started server untouched)."""
    if _mock_server_running():
        yield
        return

    server = HTTPServer(("localhost", MOCK_SERVER_PORT), mockserver.MockResponseHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for _ in range(20):
        if _mock_server_running():
            break
        time.sleep(0.1)

    yield

    server.shutdown()
    thread.join()


# IMPORTANT: it must be run at class level as first test case depends on the the another one
@pytest.mark.integration
class TestSupervisorGraph:
    def test_full_graph_succeeds_with_reliable_evidence_on_first_try(
        self, monkeypatch
    ) -> None:
        """Runs the whole compiled graph end-to-end (define_objectives -> call_retrieval_agent ->
        call_evidence_analysis_agent -> END) against the real model and the real retrieval and
        evidence analysis sub-agents, exercising the success path where the gathered papers are
        deemed reliable on the first try and no retrieval retry is needed."""
        # Keep the integration test cheap/fast; production keeps NUM_DIMENSIONS=2.
        monkeypatch.setattr(retrieval_agent, "NUM_SEARCH_DIMENSIONS", 1)
        monkeypatch.setattr(search_api, "SEMANTIC_SCHOLAR_API_URL", search_api.MOCK_SEMANTIC_SCHOLAR_API_URL)
        # Avoid flaky quality verdicts from the real model; only the retrieval/supervisor wiring is under test.
        # monkeypatch.setattr(evidence_analysis_agent.assess_evidence_quality, "func", 
        #                     lambda paper: evidence_analysis_agent.RELIABLE_DESCRIPTION)

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
        assert result.get("report_path")
        assert result.get("briefing")

    def test_full_graph_succeeds_with_unreliable_evidence_on_first_try_then_retry(
        self, monkeypatch
    ) -> None:
        """Runs the whole compiled graph end-to-end (define_objectives -> call_retrieval_agent ->
        call_evidence_analysis_agent -> END) against the real model and the real retrieval and
        evidence analysis sub-agents, exercising the success path where the gathered papers are
        deemed unreliable on the first try, triggering a retrieval retry, and then deemed reliable."""
        # Keep the integration test cheap/fast; production keeps NUM_DIMENSIONS=2.
        monkeypatch.setattr(retrieval_agent, "NUM_SEARCH_DIMENSIONS", 1)
        monkeypatch.setattr(search_api, "SEMANTIC_SCHOLAR_API_URL", search_api.MOCK_SEMANTIC_SCHOLAR_API_URL)
        result = supervisor_graph.invoke(
            SupervisorState(
                request=(
                    "How does generative AI use among university students relate to "
                    "their academic performance?"
                )
            )
        )
        assert result
        assert "papers" in result
        assert len(result["papers"]) > 0
        assert result["quality_feedback"] == evidence_analysis_agent.RELIABLE_DESCRIPTION
        assert result.get("retry_count", 0) == 1
        assert result.get("insights")
        assert result.get("report_path")
        assert result.get("briefing")

