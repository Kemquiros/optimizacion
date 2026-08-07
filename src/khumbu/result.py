"""Return type shared by every optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Step:
    """A single iterate."""

    iteration: int
    x: float
    fx: float
    error: float


@dataclass(frozen=True)
class Result:
    """Outcome of an optimization run.

    Attributes:
        x: Best point found.
        fx: Objective value at ``x``.
        iterations: Number of iterations actually performed.
        converged: True when the stopping tolerance was met before the
            iteration budget ran out. A run that exhausts its budget is
            reported as ``False`` rather than silently accepted.
        history: Every iterate, in order.

    """

    x: float
    fx: float
    iterations: int
    converged: bool
    history: list[Step] = field(default_factory=list)
