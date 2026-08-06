"""Derivative-free and derivative-based bracketing methods."""

from __future__ import annotations

import math
from collections.abc import Callable

from numopt.result import Result, Step

_INVERSE_GOLDEN_RATIO = (math.sqrt(5.0) - 1.0) / 2.0


def golden_section(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
    maximize: bool = False,
) -> Result:
    """Golden-section search on a unimodal interval.

    The interval shrinks by the constant factor ``(sqrt(5) - 1) / 2`` per
    iteration, so exactly one new function evaluation is needed per step: the
    other interior point is reused. Convergence is linear with that ratio,
    which is the price of using no derivative information at all.

    Args:
        f: Objective, assumed unimodal on ``[a, b]``.
        a: Lower bound.
        b: Upper bound.
        tolerance: Stop once the bracket is narrower than this.
        max_iterations: Iteration budget.
        maximize: Search for a maximum instead of a minimum.

    Returns:
        The best point found, with the full bracket history.

    Raises:
        ValueError: If the interval is empty or inverted.

    """
    if not b > a:
        raise ValueError(f"require a < b, got a={a}, b={b}")

    def better(new: float, old: float) -> bool:
        """Compare two objective values in the direction being optimized."""
        return new > old if maximize else new < old

    lower, upper = a, b
    step = _INVERSE_GOLDEN_RATIO * (upper - lower)
    x1, x2 = upper - step, lower + step
    f1, f2 = f(x1), f(x2)

    history: list[Step] = []
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        width = upper - lower
        if better(f1, f2):
            upper, x2, f2 = x2, x1, f1
            x1 = upper - _INVERSE_GOLDEN_RATIO * (upper - lower)
            f1 = f(x1)
            best_x, best_f = x1, f1
        else:
            lower, x1, f1 = x1, x2, f2
            x2 = lower + _INVERSE_GOLDEN_RATIO * (upper - lower)
            f2 = f(x2)
            best_x, best_f = x2, f2
        history.append(Step(iteration, best_x, best_f, width))
        if width < tolerance:
            converged = True
            break

    midpoint = (lower + upper) / 2.0
    return Result(midpoint, f(midpoint), iteration, converged, history)


def bisection(
    df: Callable[[float], float],
    a: float,
    b: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> Result:
    """Bisection on the derivative, locating a stationary point.

    Requires a sign change of ``df`` across ``[a, b]``; the bracket is halved
    each step, so the error bound is deterministic — ``(b - a) / 2^n`` — which
    is the property that makes bisection the safe fallback when faster methods
    diverge.

    Args:
        df: Derivative of the objective.
        a: Lower bound.
        b: Upper bound.
        tolerance: Stop once the bracket is narrower than this.
        max_iterations: Iteration budget.

    Returns:
        The stationary point, with the bracket history.

    Raises:
        ValueError: If the interval is inverted, or ``df`` does not change sign
            across it — in which case bisection has no guarantee to offer and
            failing loudly is better than returning an endpoint.

    """
    if not b > a:
        raise ValueError(f"require a < b, got a={a}, b={b}")
    fa, fb = df(a), df(b)
    if fa * fb > 0:
        raise ValueError(
            f"the derivative does not change sign on the interval: df({a})={fa:g}, df({b})={fb:g}"
        )

    lower, upper = a, b
    history: list[Step] = []
    converged = False
    iteration = 0
    midpoint = (lower + upper) / 2.0
    for iteration in range(1, max_iterations + 1):
        midpoint = (lower + upper) / 2.0
        value = df(midpoint)
        width = upper - lower
        history.append(Step(iteration, midpoint, value, width))
        if width < tolerance or value == 0.0:
            converged = True
            break
        if df(lower) * value < 0:
            upper = midpoint
        else:
            lower = midpoint

    return Result(midpoint, df(midpoint), iteration, converged, history)
