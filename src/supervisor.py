"""Top-level supervisor that coordinates the calendar sub-agent.

The supervisor only sees the sub-agent's public tool (`schedule_event`) - it
has no knowledge of how the sub-agent validates or retries its own work.
"""
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

from framework import model
from retrieval_agent import manage_search
from search_tools import Paper


class SupervisorState(TypedDict):
    request: str
    objectives: NotRequired[list[str]]
    insights: NotRequired[str]
    papers: NotRequired[list[Paper]]


NUM_OBJECTIVES = 1

DEFINE_OBJECTIVES_PROMPT = (
    "You are an academic investigator assistant. Given a natural language research "
    "question, define exactly {num_objectives} research objectives, ordered by "
    "importance, needed to conduct the research on the user's research question."
).format(num_objectives=NUM_OBJECTIVES)


class ResearchObjectives(BaseModel):
    objectives: list[str] = Field(
        description=(
            f"Exactly {NUM_OBJECTIVES} research objectives, ordered descending by importance, "
            "each phrased as a short goal (e.g., 'Measure the extent of generative "
            "AI use among university students')."
        )
    )


def define_objectives(state: SupervisorState, config: RunnableConfig) -> dict:
    """Step 1: ask the LLM to break the research question into N ordered research objectives."""
    print(f"Defining research objectives for request: {state['request']}")
    objectives_model = model.with_structured_output(ResearchObjectives)
    output = objectives_model.invoke(
        [
            {"role": "system", "content": DEFINE_OBJECTIVES_PROMPT},
            {"role": "user", "content": state["request"]},
        ],
        config=config,
    )
    return {"objectives": output.objectives}

"""
Implementation decision:
Supervisor delegates all information retrieval tasks to the retrieval agent, ensuring that the 
process of gathering relevant academic papers is handled efficiently and systematically. It also makes
a clear separation of concerns between agents, allowing maintainability and extensibility.
"""
def call_retrieval_agent(state: SupervisorState, config: RunnableConfig) -> dict:
    print(f"Calling retrieval agent to gather information for {NUM_OBJECTIVES} ordered research objectives")
    """Step 2: hand the research objectives to the retrieval sub-agent as one natural language request."""
    request = "find papers on this research objectives: " + " ".join(
        f"{objective}." for objective in state.get("objectives", [])
    )
    result = manage_search.invoke({"request": request}, config=config)
    return {
        "insights": result["insights"],
        "papers": result["papers"]
    }


_graph = StateGraph(SupervisorState)
_graph.add_node("define_objectives", define_objectives)
_graph.add_node("call_retrieval_agent", call_retrieval_agent)
_graph.add_edge(START, "define_objectives")
_graph.add_edge("define_objectives", "call_retrieval_agent")
_graph.add_edge("call_retrieval_agent", END)
supervisor_graph = _graph.compile()

