# Plug in your own LLM / agent

SAGE's agent is just a **proposer** — anything that, given the current code, returns an
edited version. The default uses the OpenAI API, but swapping it out is easy. Three
levels, cheapest first.

## The contract

A proposer is any object with this one method (see `OpenAIProposer` in `local_task.py`):

```python
def propose(self, current_code: str, best_score: float,
            history: list[dict]) -> Candidate | None:
    ...
```

- `current_code` — the current best method's full source (starts as the task baseline).
- `best_score` — the incumbent's held-out score so far.
- `history` — list of past `{intent, score, accepted}` dicts.
- **Returns** a `Candidate` wrapping a **complete edited method file** (a string that
  implements the `fit`/`predict` interface in `base_method.py`), or `None` to skip.

```python
from local_task import Candidate, static_check
new_code = ...                                  # your model's edited method (str)
ok, reason = static_check(new_code)             # the coherence gate (recommended)
return Candidate(code=new_code, intent="what I changed", static_ok=ok, static_reason=reason)
```

The skeptic/coherence gates, the seed re-testing, scoring, and the replay analysis are
all downstream — they don't care how `propose` produced the code.

## 1. Use a different OpenAI model

Change the model string where the proposer is built (`run_study` in `study.py`, default
`gpt-4.1-mini`):

```python
proposer = OpenAIProposer(task, model="gpt-4o")
```

## 2. Use any OpenAI-compatible endpoint (local models, other providers)

The default client is `OpenAI()`, which honors the standard env vars — so a local server
(vLLM, Ollama, LM Studio, …) or any OpenAI-compatible gateway works with **no code change**.
In `sage/.env`:

```bash
OPENAI_BASE_URL=http://localhost:11434/v1     # e.g. Ollama
OPENAI_API_KEY=not-needed                      # any non-empty string for local servers
```
then set the model name to whatever that server serves (step 1).

## 3. A fully custom agent (any provider or logic)

Write a class with the same `propose()` signature and drop it in. Example
(`sage/my_proposer.py`):

```python
from local_task import Candidate, static_check

class MyProposer:
    def __init__(self, task):
        self.task = task        # task.baseline_code (str), tasks/<name>/background.md for context

    def propose(self, current_code, best_score, history):
        # call YOUR model / agent however you like; return a COMPLETE edited method file
        new_code = my_agent(current_code=current_code, history=history)
        ok, reason = static_check(new_code)
        return Candidate(code=new_code, intent="my edit", static_ok=ok, static_reason=reason)
```

Wire it in by replacing the `OpenAIProposer(...)` line in `run_study` (`study.py`) with
`MyProposer(task)`. Everything else — gates, replay, Pareto — is unchanged.

> The agent must return a *complete* method file each step (not a diff), implementing
> `fit(X, y, seed)` / `predict(X)` from `base_method.py`. Broken or non-conforming code is
> caught by `static_check` (the coherence gate) before it costs an evaluation.
