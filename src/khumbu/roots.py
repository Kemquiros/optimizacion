"""Newton-Raphson iteration for stationary points."""

from __future__ import annotations

from collections.abc import Callable

from khumbu.result import Result, Step


def newton_raphson(
    df: Callable[[float], float],
    d2f: Callable[[float], float],
    x0: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> Result:
    """Find a stationary point by Newton's method on the derivative.

    The iteration is ``x <- x - f'(x) / f''(x)``. Convergence is quadratic near
    a simple root, and there is no guarantee at all far from one: the method
    can oscillate or diverge. That trade — speed for safety — is why this
    module also ships :func:`~khumbu.line_search.bisection`.

    Args:
        df: First derivative of the objective.
        d2f: Second derivative of the objective.
        x0: Starting point.
        tolerance: Stop once the step is smaller than this.
        max_iterations: Iteration budget.

    Returns:
        The stationary point, with the iterate history.

    Raises:
        ZeroDivisionError: If the second derivative vanishes at an iterate. The
            original 2017 script drew a fresh random start in that case, which
            hides a real failure; raising surfaces it.

    """
    x = x0
    history: list[Step] = []
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        curvature = d2f(x)
        if curvature == 0.0:
            raise ZeroDivisionError(f"vanishing second derivative at x={x:g}")
        step = df(x) / curvature
        x_next = x - step
        error = abs(x_next - x)
        history.append(Step(iteration, x_next, df(x_next), error))
        x = x_next
        if error < tolerance:
            converged = True
            break

    return Result(x, df(x), iteration, converged, history)


def secant(
    df: Callable[[float], float],
    x0: float,
    x1: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> Result:
    """Find a stationary point without ever computing the second derivative.

    Newton needs ``f''``; the secant method estimates it from the last two
    derivative values instead:

        x_{k+1} = x_k - f'(x_k) * (x_k - x_{k-1}) / (f'(x_k) - f'(x_{k-1}))

    The order of convergence is the golden ratio, ``(1 + sqrt(5)) / 2 ≈ 1.618``
    — slower than Newton's 2, but each step costs one derivative evaluation
    instead of two. When ``f''`` is expensive, secant wins on total work even
    though it loses on iteration count, which is the trade this whole package
    keeps returning to.

    That the exponent is the same golden ratio that governs
    :func:`~khumbu.line_search.golden_section` is a genuine coincidence of two
    unrelated derivations, and one of the small pleasures of the subject.

    Args:
        df: First derivative of the objective.
        x0: First starting point.
        x1: Second starting point; must differ from ``x0``.
        tolerance: Stop once the step is smaller than this.
        max_iterations: Iteration budget.

    Returns:
        The stationary point, with the iterate history.

    Raises:
        ValueError: If the two starting points coincide.
        ZeroDivisionError: If the derivative values coincide, which flattens the
            secant line and sends the next iterate to infinity.

    """
    if x0 == x1:
        raise ValueError(f"the two starting points must differ, both were {x0}")

    previous, current = x0, x1
    df_previous, df_current = df(previous), df(current)
    history: list[Step] = []
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        denominator = df_current - df_previous
        if denominator == 0.0:
            raise ZeroDivisionError(
                f"the secant is flat at x={current:g}: f'({previous:g}) == f'({current:g})"
            )
        step = df_current * (current - previous) / denominator
        nxt = current - step
        error = abs(nxt - current)
        history.append(Step(iteration, nxt, df(nxt), error))
        previous, df_previous = current, df_current
        current, df_current = nxt, df(nxt)
        if error < tolerance:
            converged = True
            break

    return Result(current, df_current, iteration, converged, history)
