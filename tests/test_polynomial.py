"""Tests for the polynomial type."""

import pytest

from khumbu import Polynomial


def test_horner_evaluation_matches_direct_powers() -> None:
    p = Polynomial([-12.0, 8.0, -1.0])  # -x^2 + 8x - 12
    for x in (-3.0, 0.0, 2.5, 10.0):
        assert p(x) == pytest.approx(-(x**2) + 8 * x - 12)


def test_derivative_is_exact() -> None:
    p = Polynomial([-12.0, 8.0, -1.0])
    dp = p.derivative()
    assert list(dp.coefficients) == [8.0, -2.0]


def test_constant_differentiates_to_zero() -> None:
    assert Polynomial([7.0]).derivative()(3.0) == 0.0


def test_degree_ignores_trailing_zeros() -> None:
    assert Polynomial([1.0, 2.0, 0.0, 0.0]).degree == 1


def test_empty_polynomial_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one coefficient"):
        Polynomial([])


def test_string_form_is_descending_and_readable() -> None:
    assert str(Polynomial([-12.0, 8.0, -1.0])) == "-1x^2 + 8x - 12"
