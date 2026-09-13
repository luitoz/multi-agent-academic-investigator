"""Unit tests for src/calendar_agent.py.

All calls to `calendar_agent.invoke` (and `schedule_graph.invoke`) are mocked via
monkeypatch, so these tests run fast and deterministically without a live Ollama
server. Tool functions are invoked via `.func(...)` to bypass LangChain's
tool-calling/schema machinery.
"""
from langchain.messages import AIMessage, ToolMessage

import calendar_agent as calendar_agent_module
from calendar_agent import (
    MAX_SCHEDULE_RETRIES,
    create_calendar_event,
    get_available_time_slots,
    give_up_scheduling,
    retry_with_first_slot,
    schedule_event,
    schedule_graph,
    schedule_node,
    should_retry_schedule,
)


class TestCreateCalendarEvent:
    def test_returns_formatted_confirmation(self):
        result = create_calendar_event.func(
            title="Team standup",
            start_time="2024-01-15T09:00:00",
            end_time="2024-01-15T09:30:00",
            attendees=["a@example.com", "b@example.com"],
        )
        assert result == (
            "Event created: Team standup from 2024-01-15T09:00:00 to "
            "2024-01-15T09:30:00 with 2 attendees"
        )

    def test_attendee_count_substitution(self):
        result = create_calendar_event.func(
            title="Solo review",
            start_time="2024-01-15T10:00:00",
            end_time="2024-01-15T10:15:00",
            attendees=[],
        )
        assert "with 0 attendees" in result


class TestGetAvailableTimeSlots:
    def test_returns_stub_slots(self):
        result = get_available_time_slots.func(
            attendees=["a@example.com"], date="2024-01-15", duration_minutes=30
        )
        assert result == ["09:00", "14:00", "16:00"]


class TestShouldRetrySchedule:
    def test_no_tool_result_with_slots_retries(self):
        assert should_retry_schedule({"tool_result": "", "slots": "['09:00']"}) is True

    def test_tool_result_present_does_not_retry(self):
        state = {"tool_result": "Event created: ...", "slots": "['09:00']", "availability_checked": True}
        assert should_retry_schedule(state) is False

    def test_neither_present_does_not_retry(self):
        # this would require human intervention
        assert should_retry_schedule({"tool_result": "", "slots": ""}) is False

    def test_tool_result_without_slots_check_retries(self):
        # event created without ever checking availability first - can't be trusted
        assert should_retry_schedule({"tool_result": "Event created: ...", "slots": ""}) is True

    def test_tool_result_present_with_prior_availability_check_does_not_retry(self):
        # this round's response didn't re-check, but a prior retry already did
        state = {"tool_result": "Event created: ...", "slots": "", "availability_checked": True}
        assert should_retry_schedule(state) is False


class TestRetryWithFirstSlot:
    def test_appends_unavailability_sentence_with_first_slot(self):
        state = {"request": "Schedule a meeting at 10am", "slots": "['09:00', '14:00']"}
        result = retry_with_first_slot(state)
        assert result["request"] == (
            "Schedule a meeting at 10am The requested time is unavailable; "
            "use 09:00 instead."
        )

    def test_forces_availability_check_when_slots_missing(self):
        state = {"request": "Schedule a meeting at 10am", "slots": ""}
        result = retry_with_first_slot(state)
        assert result["request"] == (
            "Schedule a meeting at 10am You must call get_available_time_slots "
            "before creating the event."
        )

    def test_increments_retry_count(self):
        state = {"request": "Schedule a meeting at 10am", "slots": "", "retry_count": 2}
        result = retry_with_first_slot(state)
        assert result["retry_count"] == 3


class TestGiveUpScheduling:
    def test_reports_unchecked_availability(self):
        state = {"request": "Schedule a standup", "availability_checked": False, "slots": ""}
        result = give_up_scheduling(state)
        assert "never checked availability" in result["result"]

    def test_reports_no_slot_confirmed(self):
        state = {"request": "Schedule a standup", "availability_checked": True, "slots": "['09:00']"}
        result = give_up_scheduling(state)
        assert "no available time slot could be confirmed" in result["result"]
        assert "['09:00']" in result["result"]


class TestScheduleNode:
    def test_event_created_maps_result_tool_result_and_slots(self, monkeypatch):
        slots_message = ToolMessage(
            content=str(["09:00", "14:00"]), name="get_available_time_slots", tool_call_id="1"
        )
        event_message = ToolMessage(
            content="Event created: Standup from 2024-01-15T09:00:00 to 2024-01-15T09:30:00 with 1 attendees",
            name="create_calendar_event",
            tool_call_id="2",
        )
        final_message = AIMessage(content="Scheduled the standup for 09:00.")

        def fake_invoke(_input, config=None):
            return {"messages": [slots_message, event_message, final_message]}

        monkeypatch.setattr(calendar_agent_module.calendar_agent, "invoke", fake_invoke)

        result = schedule_node({"request": "Schedule a standup"}, {})

        assert result["result"] == "Scheduled the standup for 09:00."
        assert result["tool_result"] == event_message.content
        assert result["slots"] == slots_message.content

    def test_no_event_created_only_slots(self, monkeypatch):
        slots_message = ToolMessage(
            content=str(["09:00", "14:00"]), name="get_available_time_slots", tool_call_id="1"
        )
        final_message = AIMessage(content="That time is unavailable.")

        def fake_invoke(_input, config=None):
            return {"messages": [slots_message, final_message]}

        monkeypatch.setattr(calendar_agent_module.calendar_agent, "invoke", fake_invoke)

        result = schedule_node({"request": "Schedule a standup at 10am"}, {})

        assert result["tool_result"] == ""
        assert result["slots"] == slots_message.content


class TestScheduleGraphRetryLoop:
    def test_retries_once_then_succeeds(self, monkeypatch):
        slots_message = ToolMessage(
            content=str(["09:00", "14:00"]), name="get_available_time_slots", tool_call_id="1"
        )
        first_final_message = AIMessage(content="That time is unavailable.")

        event_message = ToolMessage(
            content="Event created: Standup from 2024-01-15T09:00:00 to 2024-01-15T09:30:00 with 1 attendees",
            name="create_calendar_event",
            tool_call_id="2",
        )
        second_final_message = AIMessage(content="Scheduled the standup for 09:00.")

        responses = [
            {"messages": [slots_message, first_final_message]},
            {"messages": [event_message, second_final_message]},
        ]
        call_count = {"n": 0}

        def fake_invoke(_input, config=None):
            response = responses[call_count["n"]]
            call_count["n"] += 1
            return response

        monkeypatch.setattr(calendar_agent_module.calendar_agent, "invoke", fake_invoke)

        final_state = schedule_graph.invoke({"request": "Schedule a standup at 10am"})

        assert call_count["n"] == 2
        assert final_state["tool_result"] == event_message.content

    def test_no_retry_needed_terminates_immediately(self, monkeypatch):
        slots_message = ToolMessage(
            content=str(["09:00", "14:00"]), name="get_available_time_slots", tool_call_id="1"
        )
        event_message = ToolMessage(
            content="Event created: Standup from 2024-01-15T09:00:00 to 2024-01-15T09:30:00 with 1 attendees",
            name="create_calendar_event",
            tool_call_id="2",
        )
        final_message = AIMessage(content="Scheduled the standup for 09:00.")

        call_count = {"n": 0}

        def fake_invoke(_input, config=None):
            call_count["n"] += 1
            return {"messages": [slots_message, event_message, final_message]}

        monkeypatch.setattr(calendar_agent_module.calendar_agent, "invoke", fake_invoke)

        final_state = schedule_graph.invoke({"request": "Schedule a standup"})

        assert call_count["n"] == 1
        assert final_state["tool_result"] == event_message.content

    def test_retries_when_event_created_without_checking_availability(self, monkeypatch):
        """If the agent creates the event without ever calling get_available_time_slots,
        the graph must not accept it and should force a re-check."""
        unchecked_event_message = ToolMessage(
            content="Event created: Standup from 2024-01-15T09:00:00 to 2024-01-15T09:30:00 with 1 attendees",
            name="create_calendar_event",
            tool_call_id="1",
        )
        first_final_message = AIMessage(content="Scheduled the standup for 09:00.")

        slots_message = ToolMessage(
            content=str(["09:00", "14:00"]), name="get_available_time_slots", tool_call_id="2"
        )
        checked_event_message = ToolMessage(
            content="Event created: Standup from 2024-01-15T09:00:00 to 2024-01-15T09:30:00 with 1 attendees",
            name="create_calendar_event",
            tool_call_id="3",
        )
        second_final_message = AIMessage(content="Scheduled the standup for 09:00.")

        responses = [
            {"messages": [unchecked_event_message, first_final_message]},
            {"messages": [slots_message, checked_event_message, second_final_message]},
        ]
        call_count = {"n": 0}

        def fake_invoke(_input, config=None):
            response = responses[call_count["n"]]
            call_count["n"] += 1
            return response

        monkeypatch.setattr(calendar_agent_module.calendar_agent, "invoke", fake_invoke)

        final_state = schedule_graph.invoke({"request": "Schedule a standup"})

        assert call_count["n"] == 2
        assert final_state["tool_result"] == checked_event_message.content
        assert final_state["availability_checked"] is True

    def test_gives_up_after_max_retries_when_availability_never_checked(self, monkeypatch):
        """A stubborn agent that always creates the event without checking availability
        must not loop forever - it should give up after MAX_SCHEDULE_RETRIES."""
        unchecked_event_message = ToolMessage(
            content="Event created: Standup from 2024-01-15T09:00:00 to 2024-01-15T09:30:00 with 1 attendees",
            name="create_calendar_event",
            tool_call_id="1",
        )
        final_message = AIMessage(content="Scheduled the standup for 09:00.")

        call_count = {"n": 0}

        def fake_invoke(_input, config=None):
            call_count["n"] += 1
            return {"messages": [unchecked_event_message, final_message]}

        monkeypatch.setattr(calendar_agent_module.calendar_agent, "invoke", fake_invoke)

        final_state = schedule_graph.invoke({"request": "Schedule a standup"})

        assert call_count["n"] == MAX_SCHEDULE_RETRIES + 1
        assert final_state["availability_checked"] is False
        assert "never checked availability" in final_state["result"]

    def test_gives_up_after_max_retries_when_no_slot_ever_works(self, monkeypatch):
        """An agent that keeps checking availability but never manages to create the event
        must not loop forever - it should give up after MAX_SCHEDULE_RETRIES."""
        slots_message = ToolMessage(
            content=str(["09:00", "14:00"]), name="get_available_time_slots", tool_call_id="1"
        )
        final_message = AIMessage(content="That time is unavailable.")

        call_count = {"n": 0}

        def fake_invoke(_input, config=None):
            call_count["n"] += 1
            return {"messages": [slots_message, final_message]}

        monkeypatch.setattr(calendar_agent_module.calendar_agent, "invoke", fake_invoke)

        final_state = schedule_graph.invoke({"request": "Schedule a standup at 10am"})

        assert call_count["n"] == MAX_SCHEDULE_RETRIES + 1
        assert "no available time slot could be confirmed" in final_state["result"]


class TestScheduleEventTool:
    def test_returns_graph_result_and_forwards_request(self, monkeypatch):
        captured = {}

        def fake_invoke(input_, config=None):
            captured["input"] = input_
            return {"result": "Scheduled the standup for 09:00."}

        monkeypatch.setattr(calendar_agent_module.schedule_graph, "invoke", fake_invoke)

        result = schedule_event.func(request="Schedule a standup", config={})

        assert result == "Scheduled the standup for 09:00."
        assert captured["input"] == {"request": "Schedule a standup"}
