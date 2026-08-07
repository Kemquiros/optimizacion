"""Chapter IV — the optimisers proposed after Adam, and what they are worth.

Every method here was published as an improvement on Adam. Some are. The
honest summary of the 2025 benchmark literature is narrower than the abstracts
suggest: matrix-preconditioned methods give roughly 1.3x speedup on small
models and about **1.1x at 1.2B parameters**, and *"many reported speedups of
2x simply reflect a weak baseline"* (Fantastic Pretraining Optimizers and Where
to Find Them, arXiv:2509.02046). A well-tuned AdamW is hard to beat.

That is why this chapter exists next to the benchmark: a new optimiser is a
claim, and a claim needs a baseline that someone actually tuned.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from khumbu.multivariate import VectorResult, _add, _norm, _scale

Vector = list[float]
Matrix = list[list[float]]


def adamw(
    f: Callable[[Sequence[float]], float],
    gradient: Callable[[Sequence[float]], Sequence[float]],
    x0: Sequence[float],
    *,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 0.01,
    epsilon: float = 1e-8,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> VectorResult:
    """Adam with weight decay decoupled from the gradient.

    The distinction is subtle and it matters. Classical L2 regularisation adds
    ``λx`` to the gradient — and Adam then divides that term by ``sqrt(v)``
    along with everything else, so **the effective amount of regularisation
    depends on each coordinate's gradient history**. A rarely-updated parameter
    is decayed far more than a busy one, which nobody intended.

    AdamW applies the decay directly to the parameter instead:

        x ← x − lr·(m̂ / (√v̂ + ε) + λx)

    so it is uniform. This one change is why AdamW, not Adam, is the baseline
    that later work has to beat — and, per the 2025 benchmarks, mostly does not.

    Args:
        f: Objective.
        gradient: Gradient of the objective.
        x0: Starting point.
        learning_rate: Base step size.
        beta1: Decay for the gradient average.
        beta2: Decay for the squared-gradient average.
        weight_decay: The coefficient applied directly to the parameters.
        epsilon: Guard against division by zero.
        tolerance: Stop once the step norm falls below this.
        max_iterations: Iteration budget.

    Returns:
        The final iterate.

    Raises:
        ValueError: If a decay lies outside ``[0, 1)``, or the learning rate or
            weight decay is negative.

    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    for name, beta in (("beta1", beta1), ("beta2", beta2)):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"{name} must lie in [0, 1), got {beta}")

    x = list(x0)
    first = [0.0] * len(x)
    second = [0.0] * len(x)
    history: list[float] = []
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        g = list(gradient(x))
        first = [beta1 * m + (1 - beta1) * gi for m, gi in zip(first, g, strict=True)]
        second = [beta2 * v + (1 - beta2) * gi * gi for v, gi in zip(second, g, strict=True)]
        corrected_first = [m / (1 - beta1**iteration) for m in first]
        corrected_second = [v / (1 - beta2**iteration) for v in second]
        step = [
            -learning_rate * (m / (math.sqrt(v) + epsilon) + weight_decay * xi)
            for m, v, xi in zip(corrected_first, corrected_second, x, strict=True)
        ]
        x = _add(x, step)
        history.append(f(x))
        if _norm(step) < tolerance:
            converged = True
            break

    return VectorResult(x, f(x), iteration, iteration, converged, history)


def lion(
    f: Callable[[Sequence[float]], float],
    gradient: Callable[[Sequence[float]], Sequence[float]],
    x0: Sequence[float],
    *,
    learning_rate: float = 0.0001,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.0,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> VectorResult:
    """Lion: every step has the same length, only the direction is learned.

    Discovered by symbolic program search rather than derived (Chen et al.,
    2023). The update is the **sign** of an interpolated momentum,

        u = sign(β₁·m + (1 − β₁)·g),    x ← x − lr·(u + λx)
        m ← β₂·m + (1 − β₂)·g

    Two consequences follow from the sign, and they are the whole method:

    * **Half the memory of Adam.** One momentum buffer instead of two, because
      no second moment is tracked. At the scale where optimiser state competes
      with the model for memory, that is the entire argument.
    * **Every coordinate moves by exactly ``lr``.** Step size is completely
      decoupled from gradient magnitude, which makes Lion insensitive to
      gradient scale and *very* sensitive to the learning rate. Published
      recipes use roughly a tenth of Adam's.

    Note the asymmetry, which is easy to misread: ``β₁`` weights the *update*
    and ``β₂`` the *state*. They are not the roles they play in Adam.

    Args:
        f: Objective.
        gradient: Gradient of the objective.
        x0: Starting point.
        learning_rate: Step size — and, because of the sign, the exact distance
            each coordinate moves per step.
        beta1: Interpolation weight for the update direction.
        beta2: Decay for the momentum state.
        weight_decay: Decoupled weight decay.
        tolerance: Stop once the objective stops changing by more than this.
        max_iterations: Iteration budget.

    Returns:
        The final iterate.

    Raises:
        ValueError: If a coefficient lies outside ``[0, 1)`` or the learning
            rate is not positive.

    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    for name, beta in (("beta1", beta1), ("beta2", beta2)):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"{name} must lie in [0, 1), got {beta}")

    x = list(x0)
    momentum_state = [0.0] * len(x)
    history: list[float] = []
    previous = f(x)
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        g = list(gradient(x))
        update = [
            math.copysign(1.0, beta1 * m + (1 - beta1) * gi)
            if (beta1 * m + (1 - beta1) * gi) != 0
            else 0.0
            for m, gi in zip(momentum_state, g, strict=True)
        ]
        x = [
            xi - learning_rate * (ui + weight_decay * xi) for xi, ui in zip(x, update, strict=True)
        ]
        momentum_state = [
            beta2 * m + (1 - beta2) * gi for m, gi in zip(momentum_state, g, strict=True)
        ]
        value = f(x)
        history.append(value)
        if abs(previous - value) < tolerance:
            converged = True
            break
        previous = value

    return VectorResult(x, f(x), iteration, iteration, converged, history)


def sharpness_aware(
    f: Callable[[Sequence[float]], float],
    gradient: Callable[[Sequence[float]], Sequence[float]],
    x0: Sequence[float],
    *,
    learning_rate: float = 0.01,
    radius: float = 0.05,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> VectorResult:
    """SAM: minimise the worst point in a neighbourhood, not the point itself.

    Two minima with the same value are not equally good. A sharp one sits at
    the bottom of a narrow crevasse, so a small shift in the data moves the
    loss a lot; a flat one tolerates the shift. Sharpness-aware minimisation
    (Foret et al., 2021) optimises the *worst value within a ball of radius ρ*
    rather than the value at the point:

        ε = ρ·∇f(x)/‖∇f(x)‖        climb to the worst nearby point
        x ← x − lr·∇f(x + ε)        descend using the gradient measured there

    It costs **two gradient evaluations per step** — the price of asking a
    harder question — and the benchmark in this package charges it accordingly.

    Args:
        f: Objective.
        gradient: Gradient of the objective.
        x0: Starting point.
        learning_rate: Step size for the descent half of the step.
        radius: The neighbourhood radius ρ.
        tolerance: Stop once the step norm falls below this.
        max_iterations: Iteration budget.

    Returns:
        The final iterate.

    Raises:
        ValueError: If the radius or learning rate is not positive.

    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    x = list(x0)
    history: list[float] = []
    evaluations = 0
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        g = list(gradient(x))
        norm = _norm(g)
        if norm == 0.0:
            converged = True
            break
        ascent = _scale(g, radius / norm)
        g_worst = list(gradient(_add(x, ascent)))
        evaluations += 2
        step = _scale(g_worst, -learning_rate)
        x = _add(x, step)
        history.append(f(x))
        if _norm(step) < tolerance:
            converged = True
            break

    return VectorResult(x, f(x), iteration, evaluations, converged, history)


def _newton_schulz(matrix: Matrix, steps: int = 5) -> Matrix:
    """Approximate the orthogonal factor of a matrix, without an SVD.

    Applies the quintic iteration ``X ← aX + bX(XᵀX) + cX(XᵀX)²`` with the
    coefficients used in the reference Muon implementation. It pulls the
    singular values toward one without ever computing them, which is what makes
    the method affordable: an SVD per step would cost more than the training it
    accelerates.

    **It does not converge to an orthogonal matrix, and that is deliberate.**
    The coefficients are tuned so the singular values land in a band around one
    -- roughly [0.7, 1.3] -- as fast as possible, and iterating further makes
    the result oscillate inside that band rather than sharpen. Muon does not
    need exactness; it needs no direction to dominate. Paying for an accurate
    orthogonalisation here would buy nothing and cost the speedup.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    rows, cols = len(matrix), len(matrix[0])
    scale = math.sqrt(sum(v * v for row in matrix for v in row)) + 1e-7
    x = [[v / scale for v in row] for row in matrix]

    for _ in range(steps):
        xt_x = [
            [sum(x[k][i] * x[k][j] for k in range(rows)) for j in range(cols)] for i in range(cols)
        ]
        x_xt_x = [
            [sum(x[i][k] * xt_x[k][j] for k in range(cols)) for j in range(cols)]
            for i in range(rows)
        ]
        xt_x_2 = [
            [sum(xt_x[i][k] * xt_x[k][j] for k in range(cols)) for j in range(cols)]
            for i in range(cols)
        ]
        x_xt_x_2 = [
            [sum(x[i][k] * xt_x_2[k][j] for k in range(cols)) for j in range(cols)]
            for i in range(rows)
        ]
        x = [
            [a * x[i][j] + b * x_xt_x[i][j] + c * x_xt_x_2[i][j] for j in range(cols)]
            for i in range(rows)
        ]
    return x


def muon(
    f: Callable[[Matrix], float],
    gradient: Callable[[Matrix], Matrix],
    x0: Matrix,
    *,
    learning_rate: float = 0.02,
    decay: float = 0.95,
    newton_schulz_steps: int = 5,
    max_iterations: int = 1000,
) -> VectorResult:
    """Muon: orthogonalise the momentum before stepping.

    Unlike everything else in this package, Muon treats a parameter as a
    **matrix** rather than a flat vector, and that is the point. A gradient
    matrix is usually dominated by a few large singular directions, so plain
    momentum keeps pushing along the same handful of axes. Muon replaces the
    momentum by its nearest orthogonal matrix — every singular value set to one
    — so all directions advance at a comparable rate.

    The orthogonalisation uses a Newton-Schulz iteration rather than an SVD,
    which is what keeps it affordable.

    **What it is actually worth.** Roughly 1.3x fewer steps than a tuned AdamW
    below ~520M parameters, decaying to about 1.1x at 1.2B, at 1.45x the wall
    time per step (arXiv:2509.02046). Whether that is a win depends on the
    scale you work at — which is why this package makes you measure it.

    Args:
        f: Objective taking a matrix.
        gradient: Gradient, returned as a matrix of the same shape.
        x0: Starting matrix.
        learning_rate: Step size.
        decay: Momentum coefficient.
        newton_schulz_steps: Iterations of the orthogonalisation.
        max_iterations: Iteration budget.

    Returns:
        The final iterate, flattened row-major into ``x``.

    Raises:
        ValueError: If the starting matrix is empty or ragged.

    """
    if not x0 or not x0[0]:
        raise ValueError("the starting matrix must be non-empty")
    if any(len(row) != len(x0[0]) for row in x0):
        raise ValueError("the starting matrix must be rectangular")

    rows, cols = len(x0), len(x0[0])
    x = [list(row) for row in x0]
    velocity = [[0.0] * cols for _ in range(rows)]
    history: list[float] = []

    for _ in range(max_iterations):
        g = gradient(x)
        velocity = [[decay * velocity[i][j] + g[i][j] for j in range(cols)] for i in range(rows)]
        direction = _newton_schulz(velocity, newton_schulz_steps)
        x = [[x[i][j] - learning_rate * direction[i][j] for j in range(cols)] for i in range(rows)]
        history.append(f(x))

    flat = [value for row in x for value in row]
    return VectorResult(flat, f(x), max_iterations, max_iterations, False, history)
