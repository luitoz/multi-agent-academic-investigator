"""Generic LangGraph plumbing shared by the retry-capable agent graphs.

This module has no knowledge of calendars, emails, or any other domain concept -
it only knows how to wire up a "run an agent, validate its tool output, retry if
needed" loop, and how to pull an agent's ground-truth tool output out of its
message history.
"""
from typing import Callable

from langchain.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

model = ChatOllama(
    model="qwen3:14b",
    temperature=0,
    base_url="http://localhost:11434",
    reasoning=False,  # qwen3's chain-of-thought tokens make every agent hop far slower otherwise
    # other params...
)


def extract_tool_result(messages: list, tool_name: str) -> str:
    """Return the content of the most recent `tool_name` ToolMessage, or "" if it was never called.

    An agent's paraphrased final text is unreliable for validation; the tool's
    own output is the ground truth for whether an action actually happened.
    """
    return next(
        (m.content for m in reversed(messages) if isinstance(m, ToolMessage) and m.name == tool_name),
        "",
    )


