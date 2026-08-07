"""Secant, Brent and the Armijo line search."""

import math

import pytest

from khumbu import Polynomial, backtracking, brent, secant


def test_secant_finds_the_stationary_point_without_a_second_derivative() -> None:
    p = Polynomial([-12.0, 8.0, -1.0])
    result = secant(p.derivative(), 0.0, 1.0)
    assert result.converged
    assert result.x == pytest.approx(4.0, abs=1e-9)


def test_secant_needs_more_steps_than_newton_but_no_second_derivative() -> None:
    p = Polynomial([16.0, -8.0, 1.0])  # (x - 4)^2
    result = secant(p.derivative(), 0.0, 1.0)
    # Superlinear, not quadratic: it does not land exactly on the first step.
    assert result.iterations >= 2
    assert result.x == pytest.approx(4.0, abs=1e-9)


def test_secant_rejects_identical_starting_points() -> None:
    with pytest.raises(ValueError, match="must differ"):
        secant(lambda x: x, 1.0, 1.0)


def test_secant_surfaces_a_flat_secant() -> None:
    with pytest.raises(ZeroDivisionError, match="secant is flat"):
        secant(lambda x: 5.0, 0.0, 1.0)


def test_brent_matches_the_known_minimum() -> None:
    result = brent(lambda x: (x - 3.0) ** 2, -10.0, 10.0)
    assert result.converged
    assert result.x == pytest.approx(3.0, abs=1e-6)


def test_brent_beats_golden_section_on_a_smooth_function() -> None:
    from khumbu import golden_section

    smooth = Polynomial([16.0, -8.0, 1.0])
    fast = brent(smooth, 0.0, 10.0, tolerance=1e-10)
    slow = golden_section(smooth, 0.0, 10.0, tolerance=1e-10)
    assert fast.iterations < slow.iterations


def test_brent_handles_a_transcendental_objective() -> None:
    result = brent(lambda x: math.sin(x) + x * x / 10, -5.0, 5.0)
    assert result.converged
    # f'(x) = cos(x) + x/5 vanishes near x = -1.3064; the value there is -0.7946.
    assert result.x == pytest.approx(-1.30644, abs=1e-4)
    assert result.fx == pytest.approx(-0.794582, abs=1e-5)


def test_brent_rejects_an_inverted_interval() -> None:
    with pytest.raises(ValueError, match="require a < b"):
        brent(lambda x: x, 5.0, 1.0)


def test_armijo_returns_a_step_that_actually_decreases_the_objective() -> None:
    f, df = lambda x: (x - 3.0) ** 2, lambda x: 2.0 * (x - 3.0)
    x = 0.0
    direction = -df(x)
    step = backtracking(f, df, x, direction)
    assert f(x + step * direction) < f(x)


def test_armijo_satisfies_the_sufficient_decrease_condition() -> None:
    f, df = lambda x: (x - 3.0) ** 2, lambda x: 2.0 * (x - 3.0)
    x, c = 0.0, 1e-4
    direction = -df(x)
    step = backtracking(f, df, x, direction, sufficient_decrease=c)
    assert f(x + step * direction) <= f(x) + c * step * df(x) * direction


def test_armijo_refuses_a_direction_that_points_uphill() -> None:
    # At x = 0 the slope of (x - 3)^2 is -6, so moving in +x goes DOWN and
    # moving in -x goes up. The uphill direction here is -1, not +1.
    f, df = lambda x: (x - 3.0) ** 2, lambda x: 2.0 * (x - 3.0)
    with pytest.raises(ValueError, match="not a descent direction"):
        backtracking(f, df, 0.0, direction=-1.0)


def test_armijo_rejects_parameters_outside_their_range() -> None:
    f, df = lambda x: x * x, lambda x: 2.0 * x
    with pytest.raises(ValueError, match="shrink must lie"):
        backtracking(f, df, 1.0, -1.0, shrink=1.5)
