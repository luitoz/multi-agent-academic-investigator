from langchain.tools import tool
from langchain.chat_models import init_chat_model

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# model = init_chat_model(
#     "gemini-3.7-flash",
#     temperature=0
# )


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

from langchain.messages import SystemMessage


def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""

    response = model_with_tools.invoke(
        [
            SystemMessage(
                content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
            )
        ]
        + state["messages"]
    )
    print(f"[llm_call] added message: {response!r}")
    return {
        "messages": [response],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

from langchain.messages import ToolMessage


def tool_node(state: MessagesState):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:  # type: ignore[attr-defined]
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        # TODO use that
        message = ToolMessage(content=observation, tool_call_id=tool_call["id"])
        print(f"[tool_node] added message: {message!r}")
        result.append(message)
    return {"messages": result}

from typing import Literal
from langgraph.graph import StateGraph, START, END


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:  # type: ignore[attr-defined]
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END
# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Show the agent
from IPython.display import Image, display
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# Invoke, streaming the full message state after every node runs
from langchain.messages import HumanMessage
messages = [HumanMessage(content="Add 3 and 4.")]
# for step in agent.stream({"messages": messages}, stream_mode="values"):
#     print(f"--- state now has {len(step['messages'])} message(s) ---")
#     step["messages"][-1].pretty_print()

messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()