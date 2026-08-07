"""The benchmark must be honest before it is useful."""

import pytest

from khumbu.benchmark import PROBLEMS, run_benchmark
from khumbu.benchmark.runner import EVALUATION_SEEDS, format_table


def test_every_problem_states_what_it_exposes() -> None:
    for problem in PROBLEMS:
        assert problem.difficulty, f"{problem.name} must say what weakness it punishes"
        assert problem.optimum == 0.0


def test_gradients_agree_with_finite_differences() -> None:
    """A wrong gradient would make the whole benchmark meaningless."""
    step = 1e-6
    for problem in PROBLEMS:
        x = list(problem.start)
        analytic = problem.gradient(x)
        for i in range(len(x)):
            forward, backward = list(x), list(x)
            forward[i] += step
            backward[i] -= step
            numeric = (problem.f(forward) - problem.f(backward)) / (2 * step)
            assert analytic[i] == pytest.approx(numeric, rel=1e-4, abs=1e-4), (
                f"{problem.name} gradient disagrees in coordinate {i}"
            )


def test_the_protocol_uses_disjoint_seed_sets() -> None:
    from khumbu.benchmark.runner import CALIBRATION_SEEDS

    assert not set(CALIBRATION_SEEDS) & set(EVALUATION_SEEDS)


def test_benchmark_runs_and_reports_spread() -> None:
    outcomes = run_benchmark([PROBLEMS[0]])
    assert outcomes
    for outcome in outcomes:
        assert outcome.iqr >= 0.0
        assert outcome.best <= outcome.median + 1e-12
        assert outcome.evaluations > 0


def test_table_renders_every_outcome() -> None:
    outcomes = run_benchmark([PROBLEMS[0]])
    table = format_table(outcomes)
    for outcome in outcomes:
        assert outcome.method in table
