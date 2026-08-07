"""First-order descent methods."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from khumbu.result import Result, Step


def gradient_descent(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    *,
    learning_rate: float = 0.01,
    tolerance: float = 1e-8,
    max_iterations: int = 10_000,
) -> Result:
    """Steepest descent with a fixed step size.

    The iteration is ``x <- x - alpha * f'(x)``. For a function with
    ``L``-Lipschitz gradient the step must satisfy ``alpha < 2 / L`` or the
    iterates diverge; no line search is performed here, so choosing ``alpha``
    is the caller's responsibility and divergence is reported through
    ``converged=False`` rather than hidden.

    Args:
        f: Objective.
        df: Derivative of the objective.
        x0: Starting point.
        learning_rate: Fixed step size ``alpha``.
        tolerance: Stop once the step is smaller than this.
        max_iterations: Iteration budget.

    Returns:
        The final iterate, with the full history.

    Raises:
        ValueError: If the learning rate is not positive.

    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")

    x = x0
    history: list[Step] = []
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        x_next = x - learning_rate * df(x)
        error = abs(x_next - x)
        history.append(Step(iteration, x_next, f(x_next), error))
        x = x_next
        if error < tolerance:
            converged = True
            break

    return Result(x, f(x), iteration, converged, history)


def stochastic_gradient_descent(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    *,
    learning_rate: float = 0.01,
    noise_scale: float = 0.1,
    tolerance: float = 1e-8,
    max_iterations: int = 10_000,
    seed: int | None = None,
) -> Result:
    """Descent with additive Gaussian noise on the gradient.

    Perturbing the gradient lets the iterate escape shallow local minima that
    trap :func:`gradient_descent`, at the cost of never settling exactly: the
    stationary distribution has width proportional to ``noise_scale``. The
    noise is scaled by ``1 / sqrt(iteration)`` so that it anneals away, which
    is what makes the run converge at all.

    Args:
        f: Objective.
        df: Derivative of the objective.
        x0: Starting point.
        learning_rate: Fixed step size.
        noise_scale: Standard deviation of the perturbation at iteration one.
        tolerance: Stop once the step is smaller than this.
        max_iterations: Iteration budget.
        seed: Seed for the perturbations. Pass one to make a run reproducible;
            an unseeded stochastic result cannot be checked by anyone else.

    Returns:
        The final iterate, with the full history.

    Raises:
        ValueError: If the learning rate is not positive or the noise scale is
            negative.

    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if noise_scale < 0:
        raise ValueError(f"noise_scale must be non-negative, got {noise_scale}")

    rng = random.Random(seed)
    x = x0
    history: list[Step] = []
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        perturbation = rng.gauss(0.0, noise_scale / (iteration**0.5))
        x_next = x - learning_rate * (df(x) + perturbation)
        error = abs(x_next - x)
        history.append(Step(iteration, x_next, f(x_next), error))
        x = x_next
        if error < tolerance:
            converged = True
            break

    return Result(x, f(x), iteration, converged, history)


def conjugate_gradient(
    matrix: Sequence[Sequence[float]],
    rhs: Sequence[float],
    x0: Sequence[float] | None = None,
    *,
    tolerance: float = 1e-10,
    max_iterations: int | None = None,
) -> tuple[list[float], int, bool]:
    """Solve ``A x = b`` for symmetric positive-definite ``A``.

    Equivalent to minimizing the quadratic form ``0.5 x^T A x - b^T x``. Search
    directions are conjugate with respect to ``A``, so in exact arithmetic the
    method terminates in at most ``n`` steps — the property that separates it
    from steepest descent, which can zig-zag indefinitely in an ill-conditioned
    quadratic.

    Args:
        matrix: Symmetric positive-definite matrix ``A``.
        rhs: Right-hand side ``b``.
        x0: Starting point; defaults to the zero vector.
        tolerance: Stop once the residual norm falls below this.
        max_iterations: Iteration budget; defaults to the dimension.

    Returns:
        A tuple of the solution, the iterations performed, and whether the
        residual tolerance was met.

    Raises:
        ValueError: If the shapes disagree.

    """
    n = len(rhs)
    if any(len(row) != n for row in matrix) or len(matrix) != n:
        raise ValueError("matrix must be square and agree with the right-hand side")

    budget = n if max_iterations is None else max_iterations
    x = [0.0] * n if x0 is None else list(x0)

    def multiply(vector: Sequence[float]) -> list[float]:
        return [sum(row[j] * vector[j] for j in range(n)) for row in matrix]

    def dot(u: Sequence[float], v: Sequence[float]) -> float:
        return sum(ui * vi for ui, vi in zip(u, v, strict=True))

    residual = [b - ax for b, ax in zip(rhs, multiply(x), strict=True)]
    direction = list(residual)
    residual_norm_sq = dot(residual, residual)

    for iteration in range(1, budget + 1):
        if residual_norm_sq**0.5 < tolerance:
            return x, iteration - 1, True
        a_direction = multiply(direction)
        curvature = dot(direction, a_direction)
        if curvature <= 0:
            raise ValueError("matrix is not positive definite along a search direction")
        step = residual_norm_sq / curvature
        x = [xi + step * di for xi, di in zip(x, direction, strict=True)]
        residual = [ri - step * adi for ri, adi in zip(residual, a_direction, strict=True)]
        new_norm_sq = dot(residual, residual)
        beta = new_norm_sq / residual_norm_sq
        direction = [ri + beta * di for ri, di in zip(residual, direction, strict=True)]
        residual_norm_sq = new_norm_sq

    return x, budget, residual_norm_sq**0.5 < tolerance
