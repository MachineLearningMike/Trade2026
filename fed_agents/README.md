# Parallel FED Data + Model Research Agents

This prototype uses LangGraph to run two agent-style workers in parallel:

- `fed_data_agent`: collects FED data and validates/summarizes it.
- `model_research_agent`: prepares features, benchmarks a few regression models, and reports the best candidate.

## Run

```bash
python -m fed_agents.parallel_orchestrator
```

## Notes

- If `FRED_API_KEY` is not set, the collector falls back to a synthetic but realistic FED-like time series so the prototype still runs in local/offline environments.
- The implementation uses `langchain-core` message objects and `langgraph.StateGraph` to structure the agents.
