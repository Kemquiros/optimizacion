"""Run every applicable optimiser on every problem, under one fixed budget."""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from khumbu import adam, bfgs, momentum, nelder_mead, simulated_annealing
from khumbu.benchmark.functions import PROBLEMS, Problem
from khumbu.descent import gradient_descent as descent_1d  # noqa: F401  (kept for parity)
from khumbu.frontier import adamw, lion, sharpness_aware
from khumbu.multivariate import VectorResult

BUDGET = 2000
CALIBRATION_SEEDS = range(100, 110)
EVALUATION_SEEDS = range(30)


@dataclass(frozen=True)
class Outcome:
    """What one optimiser achieved on one problem.

    Attributes:
        method: Optimiser name.
        problem: Problem name.
        median: Median final objective value over the evaluation seeds.
        iqr: Interquartile range -- the number that says whether the median
            means anything.
        best: Best seed, reported only so it can be seen how misleading it is.
        evaluations: Median objective evaluations spent.

    """

    method: str
    problem: str
    median: float
    iqr: float
    best: float
    evaluations: int


def _perturbed_start(problem: Problem, seed: int) -> list[float]:
    """Perturb the standard start, so each seed is a different run.

    Deterministic methods have no internal randomness, so without this every
    seed would give an identical run and the spread would be a fiction.
    """
    rng = random.Random(seed)
    return [x + rng.gauss(0.0, 0.1 * (abs(x) + 1.0)) for x in problem.start]


Runner = Callable[[Problem, Sequence[float], int], VectorResult]


def _runners() -> dict[str, Runner]:
    """Every optimiser, wrapped to a common signature and a common budget."""
    return {
        "gradient descent": lambda p, x, s: momentum(
            p.f, p.gradient, x, learning_rate=1e-3, decay=0.0, max_iterations=BUDGET
        ),
        "momentum": lambda p, x, s: momentum(
            p.f, p.gradient, x, learning_rate=1e-3, decay=0.9, max_iterations=BUDGET
        ),
        "nesterov": lambda p, x, s: momentum(
            p.f, p.gradient, x, learning_rate=1e-3, decay=0.9, nesterov=True, max_iterations=BUDGET
        ),
        "adam": lambda p, x, s: adam(p.f, p.gradient, x, learning_rate=0.05, max_iterations=BUDGET),
        "adamw": lambda p, x, s: adamw(
            p.f, p.gradient, x, learning_rate=0.05, weight_decay=1e-4, max_iterations=BUDGET
        ),
        "lion": lambda p, x, s: lion(
            p.f, p.gradient, x, learning_rate=0.005, max_iterations=BUDGET
        ),
        # SAM spends two gradients per step, so it gets half the steps.
        "sam": lambda p, x, s: sharpness_aware(
            p.f, p.gradient, x, learning_rate=0.01, max_iterations=BUDGET // 2
        ),
        "bfgs": lambda p, x, s: bfgs(p.f, p.gradient, x, max_iterations=BUDGET // 10),
        "nelder-mead": lambda p, x, s: nelder_mead(p.f, x, max_iterations=BUDGET),
        "annealing": lambda p, x, s: simulated_annealing(
            p.f,
            x,
            initial_temperature=10.0,
            final_temperature=1e-3,
            iterations=BUDGET,
            step_size=0.4,
            seed=s,
        ),
    }


def run_benchmark(problems: Sequence[Problem] | None = None) -> list[Outcome]:
    """Run the full grid and return one outcome per method and problem.

    Returns:
        Outcomes sorted by problem, then by median value.

    """
    problems = problems or PROBLEMS
    outcomes: list[Outcome] = []

    for problem in problems:
        for name, runner in _runners().items():
            finals: list[float] = []
            costs: list[int] = []
            for seed in EVALUATION_SEEDS:
                start = _perturbed_start(problem, seed)
                try:
                    result = runner(problem, start, seed)
                except (ValueError, ZeroDivisionError, OverflowError):
                    continue
                value = result.fx
                if value != value or value in (float("inf"), float("-inf")):
                    continue  # a diverged run is excluded and shows up as fewer seeds
                finals.append(value)
                costs.append(result.evaluations)
            if not finals:
                continue
            quantiles = statistics.quantiles(finals, n=4) if len(finals) >= 4 else [0.0, 0.0, 0.0]
            outcomes.append(
                Outcome(
                    method=name,
                    problem=problem.name,
                    median=statistics.median(finals),
                    iqr=quantiles[2] - quantiles[0],
                    best=min(finals),
                    evaluations=int(statistics.median(costs)),
                )
            )

    return sorted(outcomes, key=lambda o: (o.problem, o.median))


def format_table(outcomes: Sequence[Outcome]) -> str:
    """Render the outcomes as a markdown table, ranked within each problem."""
    lines = [
        "| Problem | Method | Median | IQR | Best seed | Evaluations |",
        "|---|---|---|---|---|---|",
    ]
    for outcome in outcomes:
        lines.append(
            f"| {outcome.problem} | {outcome.method} | {outcome.median:.4g} | "
            f"{outcome.iqr:.3g} | {outcome.best:.4g} | {outcome.evaluations} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    results = run_benchmark()
    print(format_table(results))
