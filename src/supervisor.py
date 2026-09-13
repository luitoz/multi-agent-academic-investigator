"""Top-level supervisor that coordinates the calendar sub-agent.

The supervisor only sees the sub-agent's public tool (`schedule_event`) - it
has no knowledge of how the sub-agent validates or retries its own work.
"""
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from framework import model
from calendar_agent import schedule_event

SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant. "
    "You can schedule calendar events. "
    "Break down user requests into appropriate tool calls and coordinate the results."
)

supervisor_agent = create_agent(
    model,
    tools=[schedule_event],
    system_prompt=SUPERVISOR_PROMPT,
)


class SupervisorState(TypedDict):
    request: str
    result: str


def supervisor_node(state: SupervisorState, config: RunnableConfig) -> dict:
    """Run the supervisor agent, which coordinates the calendar tool."""
    result = supervisor_agent.invoke(
        {"messages": [{"role": "user", "content": state["request"]}]},
        config=config,
    )
    return {"result": result["messages"][-1].text}

# TODO maintain for now, until creating a conditional retry logic in the supervisor graph level
_graph = StateGraph(SupervisorState)
_graph.add_node("supervisor_node", supervisor_node)
_graph.add_edge(START, "supervisor_node")
_graph.add_edge("supervisor_node", END)
supervisor_graph = _graph.compile()
