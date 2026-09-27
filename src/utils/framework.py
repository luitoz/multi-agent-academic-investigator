"""Generic LangGraph plumbing shared by the retry-capable agent graphs.
"""
from typing import Callable

from langchain.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from search_api import Paper

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


def extract_tool_results(messages: list, tool_name: str) -> list[str]:
    """Return the content of every `tool_name` ToolMessage, in call order."""
    return [m.content for m in messages if isinstance(m, ToolMessage) and m.name == tool_name]

def shallow_paper_json(obj: Paper):
    data = obj.model_dump() if hasattr(obj, "model_dump") else obj
    return {
        k: "<object>" if isinstance(v, dict)
        else "<array>" if isinstance(v, list)
        else v
        for k, v in data.items()
        if k != "abstract"  # too long to be useful in logs
    }



