# LangGraph Multi-Agent Demo

A LangGraph supervisor agent that delegates to a calendar sub-agent, backed by
a local Ollama model.

## Project layout

- `src/framework.py` — shared model config, `build_retry_graph`, `extract_tool_result`.
- `src/calendar_agent.py` — calendar sub-agent and `schedule_event` tool.
- `src/supervisor.py` — supervisor agent that routes requests to the sub-agent.
- `src/mas_demo.py` / `demo.py` — entry points for running the demo.
- `tests/` — pytest unit tests.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The demo also expects a local [Ollama](https://ollama.com) server running
with the `qwen3:14b` model (see `src/framework.py`).

## Running the demo

```bash
python demo.py
```

or via the LangGraph CLI (see `langgraph.json`):

```bash
langgraph dev
```

## Running tests

Unit tests mock all LLM/agent calls, so they run without a live Ollama server.
Integration tests (marked `integration`) call the real model and are skipped
by default (see `addopts` in `pytest.ini`).

```bash
source .venv/bin/activate
pytest
```

Run a single file or test:

```bash
pytest tests/test_calendar_agent.py
pytest tests/test_calendar_agent.py::TestScheduleNode -v
```

Run the integration tests (requires a running Ollama server):

```bash
pytest -m integration tests/test_calendar_agent_integration.py
pytest -m integration tests/test_calendar_agent_integration.py::TestScheduleGraph::test_full_retry_graph_schedules_event_when_no_duration
```

Run all tests, including integration tests (requires a running Ollama server):

```bash
pytest -m "integration or not integration"
```

pytest captures stdout by default, so `print()` output is only shown for
failing tests. Pass `-v` to see it for passing tests too:

```bash
pytest -v tests/test_calendar_agent_integration.py
```

Note: test files must be run through `pytest` (not `python
tests/test_x.py` directly) — `pytest.ini` sets `pythonpath = src` so tests can
`import calendar_agent` etc., and that setting only applies when pytest runs
the collection.

## Debugging a test

Drop into `pdb` on failure with `-s` (disables output capturing) and
`--no-cov` (avoids coverage instrumentation interfering with breakpoints):

```bash
pytest tests/test_retrieval_agent_integration.py -m integration --pdb -s -v --no-cov
```

Or add a `breakpoint()` call where you want to inspect state and run without
`--pdb`:

```bash
pytest tests/test_retrieval_agent_integration.py -m integration -s -v --no-cov
```
