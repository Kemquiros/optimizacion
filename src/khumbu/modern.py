"""The optimisers that train neural networks — and the theory underneath them.

This module is the bridge. Everything before it is nineteenth- and
twentieth-century numerical analysis; everything in it is what a deep learning
or reinforcement learning practitioner runs every day. They are the same
subject, and reading them side by side is the point of this package.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

from khumbu.multivariate import VectorResult, _add, _norm, _scale, _subtract

Vector = list[float]


def momentum(
    f: Callable[[Sequence[float]], float],
    gradient: Callable[[Sequence[float]], Sequence[float]],
    x0: Sequence[float],
    *,
    learning_rate: float = 0.01,
    decay: float = 0.9,
    nesterov: bool = False,
    tolerance: float = 1e-8,
    max_iterations: int = 10_000,
) -> VectorResult:
    """Gradient descent with memory of where it was already going.

    Plain descent on a narrow valley zig-zags: it keeps stepping across the
    valley rather than along it. Momentum accumulates a velocity,

        v ← decay·v − lr·∇f(x),    x ← x + v

    so the components that keep pointing the same way reinforce while the
    oscillating ones cancel. It is the same cure conjugate gradient applies to
    quadratics, obtained cheaply and without requiring one.

    With ``nesterov=True`` the gradient is evaluated at ``x + decay·v`` — where
    the velocity is *about* to take you — rather than where you are. Looking
    ahead lets the method brake before overshooting instead of after.

    Args:
        f: Objective.
        gradient: Gradient of the objective.
        x0: Starting point.
        learning_rate: Step size.
        decay: Momentum coefficient in ``[0, 1)``; 0 recovers plain descent.
        nesterov: Evaluate the gradient at the look-ahead point.
        tolerance: Stop once the step norm falls below this.
        max_iterations: Iteration budget.

    Returns:
        The final iterate.

    Raises:
        ValueError: If the learning rate is not positive or the decay is outside
            ``[0, 1)``, where the velocity would grow without bound.

    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must lie in [0, 1), got {decay}")

    x = list(x0)
    velocity = [0.0] * len(x)
    history: list[float] = []
    evaluations = 0
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        probe = _add(x, _scale(velocity, decay)) if nesterov else x
        g = list(gradient(probe))
        velocity = _subtract(_scale(velocity, decay), _scale(g, learning_rate))
        x = _add(x, velocity)
        evaluations += 1
        history.append(f(x))
        if _norm(velocity) < tolerance:
            converged = True
            break

    return VectorResult(x, f(x), iteration, evaluations, converged, history)


def adam(
    f: Callable[[Sequence[float]], float],
    gradient: Callable[[Sequence[float]], Sequence[float]],
    x0: Sequence[float],
    *,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> VectorResult:
    """Adam: per-coordinate step sizes from running gradient statistics.

    Two exponential moving averages are kept — of the gradient (``m``) and of
    its square (``v``) — and the update divides one by the root of the other:

        x ← x − lr · m̂ / (√v̂ + ε)

    A coordinate whose gradient is consistently large gets a *smaller* effective
    step; a rarely-active one gets a larger. That is why Adam works on problems
    where a single global learning rate cannot serve every parameter.

    **Why the bias correction exists**, which is the part most users cannot
    explain: ``m`` and ``v`` start at zero, so early on they are biased toward
    zero — at step 1, ``m`` is only ``(1 − β₁)`` of the true gradient, about a
    tenth. Dividing by ``1 − β₁ᵏ`` undoes exactly that shrinkage. Without it the
    first steps are far too small and the run wastes its early progress. Set
    ``beta1=0`` and the correction becomes a no-op, which the test suite checks.

    Args:
        f: Objective.
        gradient: Gradient of the objective.
        x0: Starting point.
        learning_rate: Base step size.
        beta1: Decay for the gradient average.
        beta2: Decay for the squared-gradient average.
        epsilon: Guard against division by zero.
        tolerance: Stop once the step norm falls below this.
        max_iterations: Iteration budget.

    Returns:
        The final iterate.

    Raises:
        ValueError: If a decay lies outside ``[0, 1)`` or the learning rate is
            not positive.

    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
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
        # Undo the shrinkage caused by starting both averages at zero.
        corrected_first = [m / (1 - beta1**iteration) for m in first]
        corrected_second = [v / (1 - beta2**iteration) for v in second]
        step = [
            -learning_rate * m / (math.sqrt(v) + epsilon)
            for m, v in zip(corrected_first, corrected_second, strict=True)
        ]
        x = _add(x, step)
        history.append(f(x))
        if _norm(step) < tolerance:
            converged = True
            break

    return VectorResult(x, f(x), iteration, iteration, converged, history)


def simulated_annealing(
    f: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    *,
    initial_temperature: float = 1.0,
    final_temperature: float = 1e-4,
    iterations: int = 5_000,
    step_size: float = 0.5,
    seed: int | None = None,
) -> VectorResult:
    """Accept worse solutions early, and stop accepting them as you cool.

    A worse candidate is taken with probability ``exp(−Δ/T)``. When ``T`` is
    high almost anything is accepted and the search roams; as ``T`` falls the
    method hardens into hill descent. Cooling is geometric, with the rate

        α = (T_final / T_initial)^{1/(n−1)}

    derived from the schedule rather than guessed — the same construction used
    in SIROA (2018), where the temperature was defined as the maximum entropy of
    the label set.

    Unlike every gradient method here, this one can leave a local minimum. It
    pays for that with no convergence guarantee whatsoever and a strong
    dependence on the schedule.

    Args:
        f: Objective.
        x0: Starting point.
        initial_temperature: Temperature at the first iteration.
        final_temperature: Temperature at the last.
        iterations: Number of proposals.
        step_size: Standard deviation of the perturbation.
        seed: Seed for reproducibility. An unseeded stochastic result is
            evidence of nothing.

    Returns:
        The best point visited — not the last, which is usually worse.

    Raises:
        ValueError: If the temperatures are not positive and decreasing.

    """
    if initial_temperature <= 0 or final_temperature <= 0:
        raise ValueError("both temperatures must be positive")
    if final_temperature >= initial_temperature:
        raise ValueError(
            f"the schedule must cool: got initial={initial_temperature}, final={final_temperature}"
        )

    rng = random.Random(seed)
    current = list(x0)
    current_value = f(current)
    best, best_value = list(current), current_value
    cooling = (final_temperature / initial_temperature) ** (1.0 / max(iterations - 1, 1))
    temperature = initial_temperature

    history: list[float] = []
    for _ in range(iterations):
        candidate = [xi + rng.gauss(0.0, step_size) for xi in current]
        candidate_value = f(candidate)
        delta = candidate_value - current_value
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            current, current_value = candidate, candidate_value
            if candidate_value < best_value:
                best, best_value = list(candidate), candidate_value
        temperature *= cooling
        history.append(best_value)

    return VectorResult(best, best_value, iterations, iterations, True, history)


def robbins_monro(
    noisy_gradient: Callable[[Sequence[float]], Sequence[float]],
    x0: Sequence[float],
    *,
    scale: float = 1.0,
    exponent: float = 0.6,
    iterations: int = 10_000,
) -> VectorResult:
    """Stochastic approximation: the theorem underneath SGD and TD learning.

    Robbins and Monro (1951) proved that a root of a function observable only
    through noisy samples can still be found, provided the step sizes satisfy

        Σ aₖ = ∞      (able to travel any distance)
        Σ aₖ² < ∞     (noise eventually averages out)

    The schedule ``aₖ = c / k^p`` meets both exactly when ``0.5 < p ≤ 1``. Those
    two conditions are the reason learning-rate decay is not a heuristic, and
    they are the same conditions that make temporal-difference learning converge
    in reinforcement learning.

    Args:
        noisy_gradient: Returns an unbiased noisy estimate of the gradient.
        x0: Starting point.
        scale: The constant ``c``.
        exponent: The exponent ``p``; must lie in ``(0.5, 1]``.
        iterations: Number of steps.

    Returns:
        The final iterate. There is no `converged` flag worth reporting: almost
        sure convergence is asymptotic, and any finite run is still moving.

    Raises:
        ValueError: If the exponent falls outside the range where the two
            Robbins-Monro conditions both hold.

    """
    if not 0.5 < exponent <= 1.0:
        raise ValueError(
            f"exponent must lie in (0.5, 1] for both conditions to hold, got {exponent}"
        )

    x = list(x0)
    history: list[float] = []
    for k in range(1, iterations + 1):
        step = scale / (k**exponent)
        x = _subtract(x, _scale(list(noisy_gradient(x)), step))
        history.append(_norm(x))

    return VectorResult(x, float("nan"), iterations, iterations, False, history)
