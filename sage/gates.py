"""Accept policies (the "gates") and the generic experiment loop.

This module is TASK-AGNOSTIC by design: it never imports anything about machine
unlearning or the synthetic world. It operates on two callables supplied by the
caller:

    propose(rng) -> candidate            # produce one experimental change
    evaluate(candidate, seed) -> float   # noisy score, HIGHER IS BETTER, costs 1 budget unit

The same policies run unchanged against (a) the synthetic mocked eval and later
(b) the real MLRC eval. The only thing that differs between experimental ARMS is
the AcceptPolicy; the proposal stream and budget are held identical.

Budget unit = the cost of ONE full-fidelity evaluate() call. Gate overhead (extra
seeds for the causal gate, the cheap coherence check) is counted in the SAME
budget, per the "equal-budget accounting" rule in HANDOFF section 5.

COST LEVER (the `Fidelity` knob): an evaluate() call can be run at a cheaper, lower
-fidelity operating point (fewer epochs / fewer inner models / subsampled data).
A cheap eval is FASTER (costs < 1 budget unit) but NOISIER. This module stays
task-agnostic: it only knows a fidelity's `cost`; the ADAPTER interprets the rest
(`params`) and the cheaper eval's higher noise is produced by the world, not here.
Holding the budget fixed, a cheaper fidelity buys MORE evals (more seeds / steps)
at the price of per-eval precision -- the screen-cheap / confirm-expensive trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import math


# ---------------------------------------------------------------------------
# Cost lever: a fidelity is a cost/accuracy operating point for the evaluator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fidelity:
    """One operating point of the cost lever. `cost` is the budget charged per
    eval at this fidelity (full eval == 1.0). `params` is an opaque dict the TASK
    adapter interprets (e.g. {"num_models": 3} for unlearning, {"max_epochs": 3,
    "data_fraction": 0.5} for rainfall, {"sigma_mult": 1.8} for the synthetic).
    Task-agnostic code touches ONLY `name` and `cost`."""
    name: str = "full"
    cost: float = 1.0
    params: dict = field(default_factory=dict)


# Standard presets. `cost` is a PLANNED weight (validate against measured
# wall-clock on the real task); cheaper fidelities should also raise eval noise.
FULL = Fidelity(name="full", cost=1.0)
CHEAP = Fidelity(name="cheap", cost=0.3)


# ---------------------------------------------------------------------------
# Budget accounting
# ---------------------------------------------------------------------------

class Budget:
    """Counts every unit of spend. One eval seed = 1.0 unit. Coherence checks
    cost a small configurable amount so culls are not free in the accounting."""

    def __init__(self, total: float):
        self.total = float(total)
        self.spent = 0.0

    def remaining(self) -> float:
        return self.total - self.spent

    def can_afford(self, units: float) -> bool:
        return self.spent + units <= self.total + 1e-9

    def charge(self, units: float) -> None:
        self.spent += units


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    accepted: bool
    reason: str
    candidate_scores: list = field(default_factory=list)
    units_spent: float = 0.0
    delta_hat: Optional[float] = None   # estimated effect vs incumbent
    se_hat: Optional[float] = None      # estimated standard error of the effect
    culled: bool = False                # rejected by coherence before any eval


@dataclass
class Incumbent:
    """Current best. Holds cached seed scores so the causal gate can compare a
    candidate against the incumbent's own noise band without re-evaluating it."""
    scores: list = field(default_factory=list)   # measured seed scores (noisy)

    @property
    def mean(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else float("-inf")


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _mean(xs: list) -> float:
    return sum(xs) / len(xs)


def _var(xs: list) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _welch_delta_se(cand: list, base: list) -> tuple[float, float]:
    """Difference in means (cand - base) and its standard error (Welch / two
    independent samples). Returns (delta, se)."""
    delta = _mean(cand) - _mean(base)
    n_c, n_b = len(cand), len(base)
    se_sq = _var(cand) / max(n_c, 1) + _var(base) / max(n_b, 1)
    return delta, math.sqrt(max(se_sq, 0.0))


# ---------------------------------------------------------------------------
# Accept policies
# ---------------------------------------------------------------------------

class AcceptPolicy:
    name = "base"

    def decide(self, candidate, evaluate: Callable, incumbent: Incumbent,
               budget: Budget, seed0: int) -> Decision:
        raise NotImplementedError


class GreedyPolicy(AcceptPolicy):
    """The autoresearch baseline: one noisy eval, single-number comparison.
    Accept iff the candidate's single observed score beats the incumbent's
    single cached observed score. Chases noise by construction.

    `eval_cost` is the budget charged for its one eval (set from the active
    Fidelity; 1.0 = full). A cheaper fidelity lets the same budget fund more
    greedy steps, but each step's single number is noisier."""

    name = "greedy"

    def __init__(self, eval_cost: float = 1.0):
        self.eval_cost = eval_cost

    def decide(self, candidate, evaluate, incumbent, budget, seed0):
        if not budget.can_afford(self.eval_cost):
            return Decision(False, "out_of_budget", units_spent=0.0)
        s = evaluate(candidate, seed0)
        budget.charge(self.eval_cost)
        # incumbent.mean is its single cached score (greedy caches one number)
        accept = s > incumbent.mean
        delta = s - incumbent.mean
        return Decision(accept, "greedy_better" if accept else "greedy_worse",
                        candidate_scores=[s], units_spent=self.eval_cost, delta_hat=delta)


class CausalPolicy(AcceptPolicy):
    """Adaptive sequential confirmation (HANDOFF section 5, Gate 2).

    Run k0 inner seeds; estimate the paired effect vs the incumbent's cached
    seeds and its standard error. If the effect clears the noise band
    (delta - z*se > 0) accept; if it is clearly inside/negative
    (delta + z*se < 0) reject; otherwise it is borderline -> spend more seeds,
    up to k_max, then decide on the final estimate. Accept only if the gain
    clears the band. More compute-efficient than a fixed K.
    """

    name = "causal"

    def __init__(self, k0: int = 2, k_max: int = 6, z: float = 1.0, eval_cost: float = 1.0):
        self.k0 = k0
        self.k_max = k_max
        self.z = z  # ~1 SE rule by default (pre-registered)
        self.eval_cost = eval_cost  # budget per seed (from the active Fidelity)

    def decide(self, candidate, evaluate, incumbent, budget, seed0):
        scores: list = []
        k = 0
        # Always need at least 2 candidate seeds to estimate its own variance.
        while k < self.k_max:
            want = self.k0 if k == 0 else 1
            for j in range(want):
                if not budget.can_afford(self.eval_cost):
                    if len(scores) < 2:
                        return Decision(False, "out_of_budget",
                                        candidate_scores=scores,
                                        units_spent=k * self.eval_cost)
                    # decide on what we have
                    return self._finalize(scores, incumbent, k)
                scores.append(evaluate(candidate, seed0 + k))
                budget.charge(self.eval_cost)
                k += 1

            delta, se = _welch_delta_se(scores, incumbent.scores)
            if se == 0:
                # no measurable noise: decide on sign immediately
                accept = delta > 0
                return Decision(accept, "causal_zero_noise",
                                candidate_scores=scores, units_spent=k * self.eval_cost,
                                delta_hat=delta, se_hat=se)
            if delta - self.z * se > 0:
                return Decision(True, "causal_clears_band",
                                candidate_scores=scores, units_spent=k * self.eval_cost,
                                delta_hat=delta, se_hat=se)
            if delta + self.z * se < 0:
                return Decision(False, "causal_below_band",
                                candidate_scores=scores, units_spent=k * self.eval_cost,
                                delta_hat=delta, se_hat=se)
            # else borderline -> loop, spend one more seed
        return self._finalize(scores, incumbent, k)

    def _finalize(self, scores, incumbent, k):
        delta, se = _welch_delta_se(scores, incumbent.scores)
        accept = (delta - self.z * se) > 0
        return Decision(accept, "causal_final_clears" if accept else "causal_final_inconclusive",
                        candidate_scores=scores, units_spent=k * self.eval_cost,
                        delta_hat=delta, se_hat=se)


class CoherenceWrapper(AcceptPolicy):
    """Gate 1 (pre-eval, cheap). A cheap check culls broken / intent-mismatched
    candidates BEFORE any expensive eval. On a cull we charge only the small
    coherence cost and skip the inner policy entirely. Composable: wraps any
    post-eval policy (greedy or causal).

    `is_broken(candidate) -> bool` is supplied by the task. In the synthetic
    world it reads a ground-truth flag; on the real task it is a parse/import
    check plus a cheap LLM consistency check.
    """

    def __init__(self, inner: AcceptPolicy, is_broken: Callable, check_cost: float = 0.05):
        self.inner = inner
        self.is_broken = is_broken
        self.check_cost = check_cost
        self.name = f"coh+{inner.name}"

    def decide(self, candidate, evaluate, incumbent, budget, seed0):
        budget.charge(self.check_cost)  # the cheap check is not free
        if self.is_broken(candidate):
            return Decision(False, "coherence_cull", units_spent=self.check_cost,
                            culled=True)
        d = self.inner.decide(candidate, evaluate, incumbent, budget, seed0)
        d.units_spent += self.check_cost
        return d


# ---------------------------------------------------------------------------
# The generic loop
# ---------------------------------------------------------------------------

@dataclass
class StepLog:
    step: int
    accepted: bool
    reason: str
    culled: bool
    units_spent: float
    budget_spent_after: float
    delta_hat: Optional[float]
    se_hat: Optional[float]
    truth: Optional[dict] = None   # ground-truth payload (synthetic only)


def run_loop(propose: Callable, evaluate: Callable, policy: AcceptPolicy,
             budget: Budget, on_accept: Callable, init_incumbent_scores: list,
             rng, seed_counter_start: int = 10_000) -> list[StepLog]:
    """Drive propose -> gate -> (accept/revert) until the budget is exhausted.

    - on_accept(candidate, decision) is called by the caller's world to commit
      the change (e.g. advance the synthetic true-performance, or git-keep).
    - init_incumbent_scores seeds the incumbent's cached scores (the baseline).
    """
    incumbent = Incumbent(scores=list(init_incumbent_scores))
    logs: list[StepLog] = []
    step = 0
    seed_counter = seed_counter_start

    while budget.remaining() > 1e-9:
        candidate = propose(rng)
        if candidate is None:
            break
        seed_counter += 100
        dec = policy.decide(candidate, evaluate, incumbent, budget, seed_counter)
        if dec.accepted:
            on_accept(candidate, dec)
            # incumbent adopts the candidate's measured seed scores so future
            # comparisons use a fresh (not noise-inflated) reference where the
            # policy provided several; greedy provides one.
            if dec.candidate_scores:
                incumbent.scores = list(dec.candidate_scores)
        logs.append(StepLog(
            step=step, accepted=dec.accepted, reason=dec.reason, culled=dec.culled,
            units_spent=dec.units_spent, budget_spent_after=budget.spent,
            delta_hat=dec.delta_hat, se_hat=dec.se_hat,
            truth=getattr(candidate, "truth", None),
        ))
        step += 1
        # safety: if a policy could not afford even a minimal action, stop
        if dec.units_spent == 0.0 and not dec.culled:
            break

    return logs
