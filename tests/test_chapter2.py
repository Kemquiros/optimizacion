"""Nelder-Mead and BFGS on the standard multivariate test problems."""

import pytest

from khumbu import bfgs, nelder_mead


def rosenbrock(v):
    """Evaluate the classic narrow-valley test function: minimum 0 at (1, 1)."""
    x, y = v
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2


def rosenbrock_gradient(v):
    x, y = v
    return [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)]


def quadratic(v):
    return sum((vi - i - 1) ** 2 for i, vi in enumerate(v))


def quadratic_gradient(v):
    return [2 * (vi - i - 1) for i, vi in enumerate(v)]


def test_nelder_mead_solves_a_quadratic_without_derivatives() -> None:
    result = nelder_mead(quadratic, [0.0, 0.0, 0.0], tolerance=1e-10)
    assert result.converged
    for i, xi in enumerate(result.x):
        assert xi == pytest.approx(i + 1, abs=1e-4)


def test_nelder_mead_crosses_the_rosenbrock_valley() -> None:
    result = nelder_mead(rosenbrock, [-1.2, 1.0], tolerance=1e-10, max_iterations=5000)
    assert result.fx < 1e-6


def test_nelder_mead_reports_its_evaluations() -> None:
    result = nelder_mead(quadratic, [0.0, 0.0], max_iterations=50)
    assert result.evaluations > result.iterations


def test_nelder_mead_rejects_an_empty_start() -> None:
    with pytest.raises(ValueError, match="at least one dimension"):
        nelder_mead(quadratic, [])


def test_bfgs_solves_a_quadratic_quickly() -> None:
    result = bfgs(quadratic, quadratic_gradient, [0.0, 0.0, 0.0])
    assert result.converged
    for i, xi in enumerate(result.x):
        assert xi == pytest.approx(i + 1, abs=1e-6)


def test_bfgs_solves_rosenbrock() -> None:
    result = bfgs(rosenbrock, rosenbrock_gradient, [-1.2, 1.0], max_iterations=500)
    assert result.x[0] == pytest.approx(1.0, abs=1e-4)
    assert result.x[1] == pytest.approx(1.0, abs=1e-4)


def test_bfgs_needs_far_fewer_iterations_than_nelder_mead() -> None:
    """The price of having a gradient, measured."""
    with_gradient = bfgs(quadratic, quadratic_gradient, [0.0, 0.0, 0.0], tolerance=1e-8)
    without = nelder_mead(quadratic, [0.0, 0.0, 0.0], tolerance=1e-8)
    assert with_gradient.iterations < without.iterations


def test_bfgs_history_never_increases() -> None:
    result = bfgs(quadratic, quadratic_gradient, [5.0, 5.0], max_iterations=50)
    assert result.history == sorted(result.history, reverse=True)
