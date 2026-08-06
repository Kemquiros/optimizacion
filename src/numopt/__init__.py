"""Classical one-dimensional numerical optimization methods.

The algorithms in this package were written for the undergraduate course
*Optimización* (Universidad de Antioquia, 2017) and later rewritten as a
tested library. Every routine returns an :class:`~numopt.result.Result`
carrying the full iterate history, so that convergence can be inspected
rather than asserted.
"""

from numopt.descent import conjugate_gradient, gradient_descent, stochastic_gradient_descent
from numopt.line_search import bisection, golden_section
from numopt.polynomial import Polynomial
from numopt.result import Result
from numopt.roots import newton_raphson

__all__ = [
    "Polynomial",
    "Result",
    "bisection",
    "conjugate_gradient",
    "golden_section",
    "gradient_descent",
    "newton_raphson",
    "stochastic_gradient_descent",
]
__version__ = "1.0.0"
