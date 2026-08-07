"""Tests for first-order descent methods."""

import pytest

from khumbu import (
    Polynomial,
    conjugate_gradient,
    gradient_descent,
    newton_raphson,
    stochastic_gradient_descent,
)


def test_gradient_descent_reaches_known_minimum() -> None:
    result = gradient_descent(
        lambda x: (x - 3.0) ** 2, lambda x: 2.0 * (x - 3.0), 0.0, learning_rate=0.1
    )
    assert result.converged
    assert result.x == pytest.approx(3.0, abs=1e-6)


def test_gradient_descent_reports_divergence_instead_of_hiding_it() -> None:
    # alpha = 1.5 exceeds 2 / L = 1 for this objective, so the iterates diverge.
    result = gradient_descent(
        lambda x: (x - 3.0) ** 2,
        lambda x: 2.0 * (x - 3.0),
        0.0,
        learning_rate=1.5,
        max_iterations=50,
    )
    assert not result.converged


def test_gradient_descent_terminates() -> None:
    # Regression test for the 2017 script, whose stopping flag was misspelled
    # (`termin` instead of `termina`), so the loop never exited.
    result = gradient_descent(
        lambda x: x**2, lambda x: 2.0 * x, 1.0, learning_rate=0.5, max_iterations=10
    )
    assert result.iterations <= 10


def test_gradient_descent_rejects_non_positive_learning_rate() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        gradient_descent(lambda x: x, lambda x: 1.0, 0.0, learning_rate=0.0)


def test_stochastic_descent_is_reproducible_under_a_seed() -> None:
    kwargs = {"learning_rate": 0.05, "noise_scale": 0.2, "seed": 7, "max_iterations": 500}
    first = stochastic_gradient_descent(lambda x: x**2, lambda x: 2.0 * x, 5.0, **kwargs)
    second = stochastic_gradient_descent(lambda x: x**2, lambda x: 2.0 * x, 5.0, **kwargs)
    assert first.x == second.x


def test_stochastic_descent_approaches_the_minimum() -> None:
    result = stochastic_gradient_descent(
        lambda x: (x - 2.0) ** 2,
        lambda x: 2.0 * (x - 2.0),
        -5.0,
        learning_rate=0.05,
        noise_scale=0.1,
        seed=1,
        max_iterations=5_000,
    )
    assert result.x == pytest.approx(2.0, abs=0.1)


def test_newton_raphson_converges_quadratically_on_a_quadratic() -> None:
    p = Polynomial([-12.0, 8.0, -1.0])
    dp = p.derivative()
    d2p = dp.derivative()
    result = newton_raphson(dp, d2p, 0.0)
    # A quadratic objective is solved by Newton in a single step: the very first
    # iterate is already exact. A second iteration is still spent confirming that
    # the step has gone to zero, which is detection cost, not convergence cost.
    assert result.history[0].x == pytest.approx(4.0, abs=1e-12)
    assert result.iterations == 2
    assert result.converged
    assert result.x == pytest.approx(4.0, abs=1e-12)


def test_newton_raphson_surfaces_vanishing_curvature() -> None:
    with pytest.raises(ZeroDivisionError, match="vanishing second derivative"):
        newton_raphson(lambda x: x, lambda x: 0.0, 1.0)


def test_conjugate_gradient_solves_in_at_most_n_steps() -> None:
    matrix = [[4.0, 1.0], [1.0, 3.0]]
    rhs = [1.0, 2.0]
    solution, iterations, converged = conjugate_gradient(matrix, rhs)
    assert converged
    assert iterations <= len(rhs)
    assert solution[0] == pytest.approx(1.0 / 11.0, abs=1e-9)
    assert solution[1] == pytest.approx(7.0 / 11.0, abs=1e-9)


def test_conjugate_gradient_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="must be square"):
        conjugate_gradient([[1.0, 2.0]], [1.0, 2.0])
