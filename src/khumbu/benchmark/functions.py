"""Standard test functions, each chosen to punish a different weakness."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

Vector = Sequence[float]


@dataclass(frozen=True)
class Problem:
    """A benchmark problem with a known optimum.

    Attributes:
        name: Short identifier.
        f: Objective.
        gradient: Gradient of the objective.
        start: Standard starting point.
        optimum: The true minimum value.
        difficulty: What this problem is designed to expose.

    """

    name: str
    f: Callable[[Vector], float]
    gradient: Callable[[Vector], list[float]]
    start: list[float]
    optimum: float
    difficulty: str


def _sphere(v: Vector) -> float:
    return sum(x * x for x in v)


def _sphere_gradient(v: Vector) -> list[float]:
    return [2 * x for x in v]


def _rosenbrock(v: Vector) -> float:
    return sum(100 * (v[i + 1] - v[i] ** 2) ** 2 + (1 - v[i]) ** 2 for i in range(len(v) - 1))


def _rosenbrock_gradient(v: Vector) -> list[float]:
    n = len(v)
    g = [0.0] * n
    for i in range(n - 1):
        g[i] += -400 * v[i] * (v[i + 1] - v[i] ** 2) - 2 * (1 - v[i])
        g[i + 1] += 200 * (v[i + 1] - v[i] ** 2)
    return g


def _ill_conditioned(v: Vector) -> float:
    """Evaluate a quadratic bowl stretched 1000:1 -- the zig-zag maker."""
    return float(sum((1000.0 ** (i / max(len(v) - 1, 1))) * x * x for i, x in enumerate(v)))


def _ill_conditioned_gradient(v: Vector) -> list[float]:
    return [2 * (1000.0 ** (i / max(len(v) - 1, 1))) * x for i, x in enumerate(v)]


def _rastrigin(v: Vector) -> float:
    """Evaluate Rastrigin: many local minima on a global bowl."""
    return 10 * len(v) + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in v)


def _rastrigin_gradient(v: Vector) -> list[float]:
    return [2 * x + 20 * math.pi * math.sin(2 * math.pi * x) for x in v]


def _beale(v: Vector) -> float:
    x, y = v
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y * y) ** 2 + (2.625 - x + x * y**3) ** 2


def _beale_gradient(v: Vector) -> list[float]:
    x, y = v
    a, b, c = 1.5 - x + x * y, 2.25 - x + x * y * y, 2.625 - x + x * y**3
    return [
        2 * a * (y - 1) + 2 * b * (y * y - 1) + 2 * c * (y**3 - 1),
        2 * a * x + 4 * b * x * y + 6 * c * x * y * y,
    ]


PROBLEMS: list[Problem] = [
    Problem(
        "sphere",
        _sphere,
        _sphere_gradient,
        [3.0, -2.0, 1.5],
        0.0,
        "the easy case: if a method fails here, it is broken",
    ),
    Problem(
        "ill-conditioned",
        _ill_conditioned,
        _ill_conditioned_gradient,
        [1.0, 1.0, 1.0],
        0.0,
        "1000:1 curvature ratio -- exposes methods without a preconditioner",
    ),
    Problem(
        "rosenbrock",
        _rosenbrock,
        _rosenbrock_gradient,
        [-1.2, 1.0],
        0.0,
        "a narrow curved valley -- exposes methods that cannot turn",
    ),
    Problem(
        "beale",
        _beale,
        _beale_gradient,
        [1.0, 1.0],
        0.0,
        "sharp walls and a flat floor -- exposes fixed step sizes",
    ),
    Problem(
        "rastrigin",
        _rastrigin,
        _rastrigin_gradient,
        [4.5, 4.5],
        0.0,
        "a lattice of local minima -- only escaping methods survive",
    ),
]
