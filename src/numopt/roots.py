"""Newton-Raphson iteration for stationary points."""

from __future__ import annotations

from collections.abc import Callable

from numopt.result import Result, Step


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
    module also ships :func:`~numopt.line_search.bisection`.

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
