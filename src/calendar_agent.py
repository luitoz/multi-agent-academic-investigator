"""Calendar sub-agent and the `schedule_event` tool that exposes it to the supervisor."""
import ast
from datetime import date

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from framework import  extract_tool_result, model


@tool
def create_calendar_event(
    title: str,
    start_time: str,       # ISO format: "2024-01-15T14:00:00"
    end_time: str,         # ISO format: "2024-01-15T15:00:00"
    attendees: list[str],  # email addresses
    location: str = ""
) -> str:
    """Create a calendar event. Requires exact ISO datetime format."""
    # Stub: In practice, this would call Google Calendar API, Outlook API, etc.
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees"


@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # ISO format: "2024-01-15"
    duration_minutes: int
) -> list[str]:
    """Check calendar availability for given attendees on a specific date."""
    # Stub: In practice, this would query calendar APIs
    return ["09:00", "14:00", "16:00"]


CALENDAR_AGENT_PROMPT = (
    f"Today's date is {date.today().isoformat()}. "
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
    "into proper ISO datetime formats. "
    "Always call get_available_time_slots to check availability before scheduling an event. "
    "If there is no suitable time slot, stop and confirm unavailability in your response. "
    "Use create_calendar_event to schedule events. "
    "Always confirm what was scheduled in your final response."
)

calendar_agent = create_agent(
    model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
)

MAX_SCHEDULE_RETRIES = 2

# state is different from the supervisor's state; it is specific to this sub-agent
# state =! messages
class ScheduleState(TypedDict):
    request: str
    result: str
    tool_result: str
    slots: str
    availability_checked: bool
    retry_count: int
    messages: list

def schedule_node(state: ScheduleState, config) -> dict:
    """Parse and schedule the event via the calendar agent."""
    # request comes from the graph invocation by the supervisor
    result = calendar_agent.invoke(
        {"messages": [{"role": "user", "content": state["request"]}]}, 
        config=config,
    )
    messages = result["messages"] # is a list
    # for the retry graph execution, only last tool result is returned by 
    # the invoke function. i.e., more tools were executed during the 
    # workflow (e.g., check availability), but they are not returned  
    tool_result = extract_tool_result(messages, "create_calendar_event")
    print(f"schedule_node: tool_result={tool_result}") 
    slots = extract_tool_result(messages, "get_available_time_slots")
    # once checked, stays checked across retries even if a later invocation doesn't re-check
    availability_checked = bool(slots) or state.get("availability_checked", False)
    return {
        "result": messages[-1].text,
        "tool_result": tool_result,
        "slots": slots,
        "availability_checked": availability_checked,
        "messages": messages,
    }


def should_retry_schedule(state: ScheduleState) -> bool:
    """Retry if scheduling failed, or if an event was created without ever checking availability."""
    tool_result = state.get("tool_result", "")
    slots = state.get("slots", "")
    if tool_result:
        # 1 retry if an event was created without ever checking availability
        return not state.get("availability_checked", False)
    #2 retry if no event was created but slots were returned (meaning the requested time was unavailable)
    return bool(slots)


def retry_with_first_slot(state: ScheduleState) -> dict:
    """Retry the request, forcing an availability check first if one never happened."""
    retry_count = state.get("retry_count", 0) + 1
    slots = state.get("slots", "")
    if not slots:
        # 1 retry forcing an availability check if one never happened
        request = f"{state['request']} You must call get_available_time_slots before creating the event."
    else:
        # 2 retry with the first available slot if the requested time was unavailable
        first_slot = ast.literal_eval(slots)[0]
        request = f"{state['request']} The requested time is unavailable; use {first_slot} instead."
    return {"request": request, "retry_count": retry_count}


def give_up_scheduling(state: ScheduleState) -> dict:
    """Stop retrying and report why the event could not be scheduled."""
    if not state.get("availability_checked", False):
        reason = "the agent never checked availability before trying to create the event"
    else:
        reason = f"no available time slot could be confirmed (checked slots: {state.get('slots', '')})"
    return {
        "result": (
            f"Could not schedule '{state['request']}' after {MAX_SCHEDULE_RETRIES} attempts: {reason}."
        )
    }


def build_schedule_graph() -> StateGraph:

    def validate(state: dict) -> str:
        if not should_retry_schedule(state):
            return END
        if state.get("retry_count", 0) >= MAX_SCHEDULE_RETRIES:
            return "give_up_scheduling"
        return "retry_with_first_slot"

    graph = StateGraph(ScheduleState)
    graph.add_node("schedule_node", schedule_node)
    graph.add_node("retry_with_first_slot", retry_with_first_slot)
    graph.add_node("give_up_scheduling", give_up_scheduling)

    graph.add_edge(START, "schedule_node")
    graph.add_conditional_edges("schedule_node", validate, ["retry_with_first_slot", "give_up_scheduling", END])
    graph.add_edge("retry_with_first_slot", "schedule_node")
    graph.add_edge("give_up_scheduling", END)

    return graph
schedule_graph = build_schedule_graph().compile()

# entry point for the supervisor to call sub-agent as a tool
@tool
def schedule_event(request: str, config) -> str:
    """Schedule calendar events using natural language.

    Use this when the user wants to create, modify, or check calendar appointments.
    Handles date/time parsing, availability checking, and event creation.

    Input: Natural language scheduling request (e.g., 'meeting with design team
    next Tuesday at 2pm')
    """
    result = schedule_graph.invoke({"request": request}, config=config)
    return result["result"]
