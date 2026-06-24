# Machine Unlearning (advanced subsystem)

The `sage/mlrc_*` files run the SAGE skeptic on the **MLRC-Bench Machine-Unlearning**
benchmark — an autonomous greedy-vs-skeptic run on a real named task, plus the
replication audit. This is the advanced track; it needs a **CUDA GPU** and the
separate [MLRC-Bench](https://github.com/yunx-z/MLRC-Bench) environment.

> ⚠️ **Read this first.** On some hardware (notably laptop MPS) the unlearning eval is
> **non-stationary** — the *same* method scores very differently across time windows
> for reasons unrelated to the method, so gate numbers can't be trusted. Confirm the
> eval is stationary on your machine before quoting any number (run `baseline_noise.py`
> in two separated windows; the means must agree within a standard deviation). See
> [`OBSERVATIONS.md`](OBSERVATIONS.md).

## What the pipeline does

```
propose (OpenAI edits methods/MyMethod.py) → evaluate (real MLRC dev eval) → accept rule → keep/discard
```

- **Eval:** `main.py -m my_method -p dev` inside the MLRC task `env/`. It
  fine-tunes/unlearns ResNet-18 on CIFAR-10 over `NUM_MODELS` inner models plus a
  membership-inference attack, and reports a *Final Score* (higher is better).
- **Accept rule** — the only thing that changes between arms; all arms share the policy
  objects in [`../sage/gates.py`](../sage/gates.py):
  - `greedy` — accept if the single observed score beats the incumbent.
  - `causal` — re-run over `k0..k_max` seeds; accept only if the mean gain clears the
    noise band (the skeptic).
  - `coh+greedy` / `coh+causal` — same, with a cheap static coherence check that culls
    broken edits before they cost an eval.

Driver: [`../sage/run_mlrc.py`](../sage/run_mlrc.py) ·
Adapter (eval + OpenAI proposer): [`../sage/mlrc_adapter.py`](../sage/mlrc_adapter.py).

## Setup

1. Set up [MLRC-Bench](https://github.com/yunx-z/MLRC-Bench) and its Machine-Unlearning
   task on a CUDA machine, following their instructions.
2. The adapter resolves the task at
   `<repo_root>/MLRC-Bench/MLAgentBench/benchmarks_base/machine_unlearning/env`
   (see `REPO_ROOT` / `ENV_DIR` in `mlrc_adapter.py`). Place or symlink MLRC-Bench at
   the repo root so this path resolves.
3. Put your `OPENAI_API_KEY` in `sage/.env` (the proposer needs it).

## Run

```bash
cd sage
# confirm the eval is stationary on this box (run twice, compare)
python baseline_noise.py
# a greedy run (the credulous baseline)
python run_mlrc.py --arm greedy  --budget 8 --fidelity full
# the skeptic
python run_mlrc.py --arm causal  --budget 8 --fidelity full
# how many of greedy's accepted wins vanish on re-test (the strongest MU result)
python replication_audit_real.py
```

A live greedy-vs-causal head-to-head **diverges** (different accepts → different
proposals → confounded). The clean paired ablation is the **replay** design used by the
local-task `study.py`; on MU, lean on the **within-arm replication audit** and treat a
live head-to-head as corroboration only.
