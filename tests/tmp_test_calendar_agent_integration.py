"""Integration tests for src/calendar_agent.py that call the real calendar agent.

Unlike test_calendar_agent.py, these tests do NOT mock `calendar_agent.invoke`,
so they exercise the live model and require a running Ollama server.
"""
from datetime import date, datetime

import pytest
from langgraph.checkpoint.memory import MemorySaver

import calendar_agent as calendar_agent_module
from calendar_agent import schedule_graph



# TODO try edge cases to cover the give up path
class TestScheduleGraph:
    # @pytest.mark.integration
    def test_full_retry_graph_schedules_event_when_no_available_slot(self) -> None:
        """Runs the whole compiled graph (schedule_node -> validate -> retry/give_up/END),
        not just a single node, against the real model."""
        result = schedule_graph.invoke(
            {"request": "Schedule a 30 minute standup starting at 17:00pm with attendees Luis and Carmen"}
        )
        today = date.today().isoformat()
        # the model doesn't always include seconds in its ISO timestamps, so match loosely
        assert result["tool_result"].startswith("Event created: Standup")
        assert f"{today}T09:00" in result["tool_result"]
        assert f"{today}T09:30" in result["tool_result"]
        assert "2 attendees" in result["tool_result"]
        assert result["availability_checked"] is True
        assert result["result"]

    @pytest.mark.integration
    def test_full_retry_graph_schedules_event_when_available_slot(self) -> None:
        """Runs the whole compiled graph (schedule_node -> validate -> retry/give_up/END),
        not just a single node, against the real model."""
        result = schedule_graph.invoke(
            {"request": "Schedule a 30 minute standup starting at 2:00pm with attendees Luis and Carmen"}
        )
        today = date.today().isoformat()
        # the model doesn't always include seconds in its ISO timestamps, so match loosely
        assert result["tool_result"].startswith("Event created: Standup")
        assert f"{today}T14:00" in result["tool_result"]
        assert f"{today}T14:30" in result["tool_result"]
        assert "2 attendees" in result["tool_result"]
        assert result["availability_checked"] is True
        assert result["result"]
