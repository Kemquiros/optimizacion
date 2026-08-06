"""Dense univariate polynomials with exact symbolic differentiation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Polynomial:
    """A univariate polynomial in ascending-degree coefficient order.

    ``Polynomial([-12.0, 8.0, -1.0])`` represents ``-x^2 + 8x - 12``.

    Coefficients are stored ascending because that makes differentiation and
    Horner evaluation index-free; the string form is printed descending, the
    way it is written by hand.
    """

    coefficients: Sequence[float]

    def __post_init__(self) -> None:
        """Reject the degenerate empty polynomial."""
        if not self.coefficients:
            raise ValueError("a polynomial needs at least one coefficient")

    @property
    def degree(self) -> int:
        """Degree, ignoring trailing zero coefficients."""
        significant = [i for i, c in enumerate(self.coefficients) if c != 0]
        return significant[-1] if significant else 0

    def __call__(self, x: float) -> float:
        """Evaluate by Horner's rule, which is numerically steadier than powers."""
        acc = 0.0
        for coefficient in reversed(self.coefficients):
            acc = acc * x + coefficient
        return acc

    def derivative(self) -> Polynomial:
        """Exact derivative. Constant polynomials differentiate to zero."""
        if len(self.coefficients) == 1:
            return Polynomial([0.0])
        return Polynomial([i * c for i, c in enumerate(self.coefficients)][1:])

    def __str__(self) -> str:
        """Render in descending degree, the way the polynomial is written by hand."""
        terms: list[str] = []
        for power, coefficient in reversed(list(enumerate(self.coefficients))):
            if coefficient == 0:
                continue
            if power == 0:
                body = f"{abs(coefficient):g}"
            elif power == 1:
                body = f"{abs(coefficient):g}x"
            else:
                body = f"{abs(coefficient):g}x^{power}"
            sign = "-" if coefficient < 0 else "+"
            terms.append(f"{sign} {body}" if terms else (f"-{body}" if coefficient < 0 else body))
        return " ".join(terms) if terms else "0"
