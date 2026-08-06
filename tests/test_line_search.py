"""Tests for bracketing methods, checked against analytically known optima."""

import itertools
import math

import pytest

from numopt import Polynomial, bisection, golden_section


def test_golden_section_finds_known_minimum() -> None:
    # (x - 3)^2 has its minimum at x = 3.
    result = golden_section(lambda x: (x - 3.0) ** 2, -10.0, 10.0)
    assert result.converged
    assert result.x == pytest.approx(3.0, abs=1e-6)


def test_golden_section_finds_known_maximum() -> None:
    # -x^2 + 8x - 12 peaks at x = 4.
    p = Polynomial([-12.0, 8.0, -1.0])
    result = golden_section(p, 0.0, 10.0, maximize=True)
    assert result.x == pytest.approx(4.0, abs=1e-6)


def test_golden_section_shrinks_the_bracket_monotonically() -> None:
    result = golden_section(lambda x: (x - 1.0) ** 2, -5.0, 5.0)
    widths = [step.error for step in result.history]
    assert widths == sorted(widths, reverse=True)


def test_golden_section_rejects_inverted_interval() -> None:
    with pytest.raises(ValueError, match="require a < b"):
        golden_section(lambda x: x, 5.0, 1.0)


def test_bisection_locates_stationary_point() -> None:
    p = Polynomial([-12.0, 8.0, -1.0])
    result = bisection(p.derivative(), 0.0, 10.0)
    assert result.converged
    assert result.x == pytest.approx(4.0, abs=1e-8)


def test_bisection_requires_a_sign_change() -> None:
    # (x - 3)^2 has derivative 2(x - 3), which is negative on all of [-5, 0].
    with pytest.raises(ValueError, match="does not change sign"):
        bisection(lambda x: 2.0 * (x - 3.0), -5.0, 0.0)


def test_bisection_error_bound_halves_each_step() -> None:
    result = bisection(lambda x: x, -1.0, 2.0, tolerance=1e-6)
    widths = [step.error for step in result.history]
    for previous, current in itertools.pairwise(widths):
        assert current == pytest.approx(previous / 2.0)


def test_golden_section_on_the_transcendental_objective_of_the_original_script() -> None:
    # The 2017 script hard-coded 2 sin(x) - x^2 / 10; its maximum near x = 1.4276
    # is the reference value that run was meant to reproduce.
    result = golden_section(lambda x: 2 * math.sin(x) - (x**2) / 10, 0.0, 4.0, maximize=True)
    assert result.x == pytest.approx(1.4276, abs=1e-3)
