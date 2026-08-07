"""Derivative-free and derivative-based bracketing methods."""

from __future__ import annotations

import math
from collections.abc import Callable

from khumbu.result import Result, Step

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


def brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> Result:
    """Brent's method: what production libraries actually use.

    Golden section is safe but slow; fitting a parabola through three points and
    jumping to its vertex is fast but can propose a step outside the bracket or
    fail to shrink it. Brent takes the parabolic step *when it is provably
    sensible* — inside the interval, and less than half the step before last —
    and falls back to a golden-section step otherwise.

    The result is superlinear convergence on smooth functions with golden
    section's guarantee retained on hostile ones. This hedge, not the
    interpolation, is the idea worth taking away: a fast method with a safe
    fallback beats either alone.

    Args:
        f: Objective, assumed unimodal on ``[a, b]``.
        a: Lower bound.
        b: Upper bound.
        tolerance: Stop once the bracket is narrower than this.
        max_iterations: Iteration budget.

    Returns:
        The minimiser, with the full history.

    Raises:
        ValueError: If the interval is empty or inverted.

    """
    if not b > a:
        raise ValueError(f"require a < b, got a={a}, b={b}")

    golden = (3.0 - math.sqrt(5.0)) / 2.0
    lower, upper = a, b
    x = w = v = lower + golden * (upper - lower)
    fx = fw = fv = f(x)
    step = previous_step = 0.0

    history: list[Step] = []
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        midpoint = 0.5 * (lower + upper)
        width = upper - lower
        history.append(Step(iteration, x, fx, width))
        if width < tolerance:
            converged = True
            break

        parabolic = False
        if abs(previous_step) > tolerance:
            # Fit a parabola through (v, fv), (w, fw), (x, fx) and take its vertex.
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0:
                p = -p
            q = abs(q)
            # Accept the parabolic step only if it stays inside the bracket and
            # is less than half the step before last: the safeguard is the method.
            if abs(p) < abs(0.5 * q * previous_step) and lower < x + p / q < upper:
                previous_step, step = step, p / q
                parabolic = True

        if not parabolic:
            previous_step = (upper - x) if x < midpoint else (lower - x)
            step = golden * previous_step

        candidate = x + step
        f_candidate = f(candidate)

        if f_candidate <= fx:
            if candidate < x:
                upper = x
            else:
                lower = x
            v, fv, w, fw, x, fx = w, fw, x, fx, candidate, f_candidate
        else:
            if candidate < x:
                lower = candidate
            else:
                upper = candidate
            if f_candidate <= fw or w == x:
                v, fv, w, fw = w, fw, candidate, f_candidate
            elif f_candidate <= fv or v in (x, w):
                v, fv = candidate, f_candidate

    return Result(x, fx, iteration, converged, history)


def backtracking(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x: float,
    direction: float,
    *,
    initial_step: float = 1.0,
    shrink: float = 0.5,
    sufficient_decrease: float = 1e-4,
    max_halvings: int = 60,
) -> float:
    """Choose a step length that guarantees progress: the Armijo condition.

    :func:`~khumbu.descent.gradient_descent` leaves the step size to the caller,
    and picking it wrongly is how that method fails. This answers the question:
    start optimistic and halve until the decrease is at least proportional to
    what the slope promised,

        f(x + t·d) ≤ f(x) + c·t·f'(x)·d

    Any ``t`` satisfying this makes real progress rather than merely moving
    downhill by an amount that shrinks faster than the step. The condition is
    what turns descent from a method that needs tuning into one that does not.

    Args:
        f: Objective.
        df: Derivative of the objective.
        x: Current point.
        direction: Search direction; must be a descent direction.
        initial_step: Step length to try first.
        shrink: Factor applied on each rejection, in ``(0, 1)``.
        sufficient_decrease: The constant ``c``, conventionally small.
        max_halvings: Give up after this many rejections.

    Returns:
        A step length satisfying the Armijo condition.

    Raises:
        ValueError: If the direction is not a descent direction, or the
            parameters are outside their valid ranges. A non-descent direction
            is a caller bug, and no step length can rescue it.

    """
    if not 0.0 < shrink < 1.0:
        raise ValueError(f"shrink must lie in (0, 1), got {shrink}")
    if not 0.0 < sufficient_decrease < 1.0:
        raise ValueError(f"sufficient_decrease must lie in (0, 1), got {sufficient_decrease}")

    slope = df(x) * direction
    if slope >= 0:
        raise ValueError(
            f"direction {direction:g} is not a descent direction at x={x:g}: "
            f"the directional derivative is {slope:g}"
        )

    baseline = f(x)
    step = initial_step
    for _ in range(max_halvings):
        if f(x + step * direction) <= baseline + sufficient_decrease * step * slope:
            return step
        step *= shrink
    return step
