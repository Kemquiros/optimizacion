"""Multivariate methods: where optimisation stops being a toy."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

Vector = list[float]


@dataclass(frozen=True)
class VectorResult:
    """Outcome of a multivariate run.

    Attributes:
        x: Best point found.
        fx: Objective value there.
        iterations: Iterations performed.
        evaluations: Objective evaluations spent — the currency that lets two
            methods be compared fairly when their iterations cost differently.
        converged: Whether the tolerance was met before the budget ran out.
        history: Best objective value after each iteration.

    """

    x: Vector
    fx: float
    iterations: int
    evaluations: int
    converged: bool
    history: list[float] = field(default_factory=list)


def _subtract(u: Sequence[float], v: Sequence[float]) -> Vector:
    return [a - b for a, b in zip(u, v, strict=True)]


def _add(u: Sequence[float], v: Sequence[float]) -> Vector:
    return [a + b for a, b in zip(u, v, strict=True)]


def _scale(u: Sequence[float], factor: float) -> Vector:
    return [factor * a for a in u]


def _dot(u: Sequence[float], v: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(u, v, strict=True))


def _norm(u: Sequence[float]) -> float:
    return math.sqrt(_dot(u, u))


def nelder_mead(
    f: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    *,
    step: float = 0.1,
    tolerance: float = 1e-8,
    max_iterations: int = 2000,
) -> VectorResult:
    """Downhill simplex: optimisation with no derivative at all.

    Keeps ``n + 1`` points — a simplex — and repeatedly replaces its worst
    vertex by reflecting it through the centroid of the others, expanding when
    the reflection pays off and contracting when it does not. The simplex
    crawls downhill and shrinks around the minimum like a tightening net.

    Use it when the objective is a black box: a simulation, an experiment, a
    piece of legacy code whose gradient nobody can derive. It has **no
    convergence guarantee** in more than one dimension — a known result, not an
    implementation flaw — and it degrades badly above roughly ten dimensions.
    Reach for :func:`bfgs` whenever a gradient exists.

    Args:
        f: Objective taking a vector.
        x0: Starting point.
        step: Size of the initial simplex around ``x0``.
        tolerance: Stop once the simplex spread falls below this.
        max_iterations: Iteration budget.

    Returns:
        The best vertex found.

    Raises:
        ValueError: If the starting point is empty.

    """
    if not len(x0):
        raise ValueError("the starting point must have at least one dimension")

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    n = len(x0)

    simplex: list[Vector] = [list(x0)]
    for i in range(n):
        vertex = list(x0)
        vertex[i] += step if vertex[i] == 0 else step * abs(vertex[i])
        simplex.append(vertex)

    values = [f(vertex) for vertex in simplex]
    evaluations = len(values)
    history: list[float] = []
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        order = sorted(range(len(simplex)), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        history.append(values[0])

        if _norm(_subtract(simplex[-1], simplex[0])) < tolerance:
            converged = True
            break

        centroid = [sum(v[j] for v in simplex[:-1]) / n for j in range(n)]
        reflected = _add(centroid, _scale(_subtract(centroid, simplex[-1]), alpha))
        f_reflected = f(reflected)
        evaluations += 1

        if values[0] <= f_reflected < values[-2]:
            simplex[-1], values[-1] = reflected, f_reflected
            continue

        if f_reflected < values[0]:
            expanded = _add(centroid, _scale(_subtract(reflected, centroid), gamma))
            f_expanded = f(expanded)
            evaluations += 1
            if f_expanded < f_reflected:
                simplex[-1], values[-1] = expanded, f_expanded
            else:
                simplex[-1], values[-1] = reflected, f_reflected
            continue

        contracted = _add(centroid, _scale(_subtract(simplex[-1], centroid), rho))
        f_contracted = f(contracted)
        evaluations += 1
        if f_contracted < values[-1]:
            simplex[-1], values[-1] = contracted, f_contracted
            continue

        # Nothing worked: shrink the whole simplex toward the best vertex.
        best = simplex[0]
        simplex = [best] + [_add(best, _scale(_subtract(v, best), sigma)) for v in simplex[1:]]
        values = [values[0]] + [f(v) for v in simplex[1:]]
        evaluations += n

    return VectorResult(simplex[0], values[0], iteration, evaluations, converged, history)


def bfgs(
    f: Callable[[Sequence[float]], float],
    gradient: Callable[[Sequence[float]], Sequence[float]],
    x0: Sequence[float],
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
) -> VectorResult:
    """Quasi-Newton: Newton's speed without ever forming a Hessian.

    Newton needs the second derivative matrix — ``n²`` entries, and a linear
    solve every step. BFGS never computes one. It *accumulates* an approximation
    of the inverse Hessian from the gradients it has already paid for, using the
    secant condition: the curvature between two points is visible in how the
    gradient changed between them.

    That is the same idea as :func:`~khumbu.roots.secant` in one dimension,
    generalised. It is why BFGS, not Newton, is what most optimisers actually
    run.

    The step length comes from an Armijo backtracking line search, so no
    learning rate needs choosing. The curvature update is skipped when it would
    destroy positive-definiteness, which is what keeps the search direction a
    descent direction.

    Args:
        f: Objective.
        gradient: Gradient of the objective.
        x0: Starting point.
        tolerance: Stop once the gradient norm falls below this.
        max_iterations: Iteration budget.

    Returns:
        The minimiser found.

    Raises:
        ValueError: If the starting point is empty.

    """
    if not len(x0):
        raise ValueError("the starting point must have at least one dimension")

    n = len(x0)
    x = list(x0)
    g = list(gradient(x))
    # Inverse-Hessian approximation, started at the identity: the first step is
    # therefore plain steepest descent, and curvature is learned from there.
    inverse_hessian = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    evaluations = 1
    history: list[float] = []
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        value = f(x)
        evaluations += 1
        history.append(value)

        if _norm(g) < tolerance:
            converged = True
            break

        direction = [-sum(inverse_hessian[i][j] * g[j] for j in range(n)) for i in range(n)]
        slope = _dot(g, direction)
        if slope >= 0:  # numerical drift: reset to steepest descent
            direction = _scale(g, -1.0)
            slope = _dot(g, direction)
            inverse_hessian = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        step_length = 1.0
        for _ in range(60):
            candidate = _add(x, _scale(direction, step_length))
            evaluations += 1
            if f(candidate) <= value + 1e-4 * step_length * slope:
                break
            step_length *= 0.5

        x_next = _add(x, _scale(direction, step_length))
        g_next = list(gradient(x_next))
        s = _subtract(x_next, x)
        y = _subtract(g_next, g)
        curvature = _dot(y, s)

        if curvature > 1e-12:  # skip the update rather than lose definiteness
            rho = 1.0 / curvature
            left = [
                [(1.0 if i == j else 0.0) - rho * s[i] * y[j] for j in range(n)] for i in range(n)
            ]
            right = [
                [(1.0 if i == j else 0.0) - rho * y[i] * s[j] for j in range(n)] for i in range(n)
            ]
            middle = [
                [sum(left[i][k] * inverse_hessian[k][j] for k in range(n)) for j in range(n)]
                for i in range(n)
            ]
            inverse_hessian = [
                [
                    sum(middle[i][k] * right[k][j] for k in range(n)) + rho * s[i] * s[j]
                    for j in range(n)
                ]
                for i in range(n)
            ]

        x, g = x_next, g_next

    return VectorResult(x, f(x), iteration, evaluations, converged, history)
