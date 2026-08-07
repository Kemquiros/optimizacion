"""khumbu — every step of the ascent, recorded.

The Khumbu icefall is the stretch every Everest ascent must cross. It is
dangerous and it is unavoidable, so the sherpas fix the route and mark it for
everyone climbing behind them.

This package does the same for optimisation. Every routine returns the *full
trail* of its run, not just the answer, and says plainly whether it converged
or merely ran out of budget. It walks from the classical methods of numerical
analysis to the optimisers that train neural networks, in one place, because
they are one subject.

Three chapters:

* **Classical** — golden section, Brent, bisection, Newton-Raphson, secant,
  and the Armijo line search that decides a step length for you.
* **Multivariate** — Nelder-Mead when there is no gradient, BFGS when there is.
* **Modern** — momentum, Nesterov, Adam, simulated annealing, and the
  Robbins-Monro theorem that underlies stochastic gradient descent and
  temporal-difference learning alike.
"""

from khumbu.descent import (
    conjugate_gradient,
    gradient_descent,
    stochastic_gradient_descent,
)
from khumbu.line_search import backtracking, bisection, brent, golden_section
from khumbu.modern import adam, momentum, robbins_monro, simulated_annealing
from khumbu.multivariate import VectorResult, bfgs, nelder_mead
from khumbu.polynomial import Polynomial
from khumbu.result import Result, Step
from khumbu.roots import newton_raphson, secant

__all__ = [
    "Polynomial",
    "Result",
    "Step",
    "VectorResult",
    "adam",
    "backtracking",
    "bfgs",
    "bisection",
    "brent",
    "conjugate_gradient",
    "golden_section",
    "gradient_descent",
    "momentum",
    "nelder_mead",
    "newton_raphson",
    "robbins_monro",
    "secant",
    "simulated_annealing",
    "stochastic_gradient_descent",
]
__version__ = "2.0.0"
