"""Regenerate the figures in the README.

    python scripts/make_figures.py

Every figure is produced from the library itself — nothing is drawn by hand,
so a change in behaviour shows up as a change in the picture.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from khumbu import Polynomial, bisection, golden_section, gradient_descent, newton_raphson  # noqa: E402

FIGURES = Path(__file__).resolve().parent.parent / "figures"
INK, ACCENT, MUTED = "#1a1a1a", "#c1440e", "#8a8a8a"
PALETTE = ["#1a1a1a", "#c1440e", "#2166ac", "#4d9221"]
# Distinct marker shapes so coincident curves stay tellable apart, and so the
# figures survive black-and-white printing and colour-blind reading.
MARKERS = ["o", "s", "D", "^"]

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.family": "serif",
        "font.size": 9,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "figure.facecolor": "white",
    }
)


def convergence_rates() -> None:
    """The central figure: how fast each method shrinks its error.

    On a logarithmic axis a straight line is linear convergence and its slope
    is the rate; Newton's curve bends downward because each step roughly
    doubles the number of correct digits.
    """
    # A single MINIMISATION objective so all four methods chase the same target:
    # (x - 4)^2, whose minimum is at x = 4. Comparing a descent method against a
    # maximisation objective would make it diverge by construction.
    objective = Polynomial([16.0, -8.0, 1.0])  # (x - 4)^2
    derivative = objective.derivative()
    second = derivative.derivative()

    runs = {
        "Golden section (no derivative)": golden_section(
            objective, 0.0, 10.0, tolerance=1e-14, max_iterations=60
        ),
        "Bisection on f'": bisection(derivative, 0.0, 10.0, tolerance=1e-14, max_iterations=60),
        "Newton-Raphson": newton_raphson(derivative, second, 0.0, tolerance=1e-14, max_iterations=60),
        "Gradient descent (a=0.3)": gradient_descent(
            objective, derivative, 0.0, learning_rate=0.3, tolerance=1e-14, max_iterations=60
        ),
    }

    figure, axes = plt.subplots(figsize=(6.6, 3.9))
    for offset, ((label, result), colour, marker) in enumerate(
        zip(runs.items(), PALETTE, MARKERS, strict=True)
    ):
        errors = [max(abs(step.x - 4.0), 1e-17) for step in result.history]
        axes.semilogy(
            range(1, len(errors) + 1), errors, label=label, color=colour, linewidth=1.6,
            marker=marker, markersize=5, markerfacecolor="white", markeredgewidth=1.2,
            markevery=(offset * 2, 6),
        )

    axes.set_xlabel("iteration")
    axes.set_ylabel(r"$|x_k - x^\ast|$")
    axes.set_title("Distance to the true optimum, same objective, same interval", loc="left")
    axes.set_xlim(1, 45)
    axes.set_ylim(1e-17, 1e1)
    axes.legend(loc="upper right")
    axes.annotate(
        "Newton: exact at the first step\n(a quadratic IS its own model)",
        xy=(1.15, 1e-16),
        xytext=(5, 1e-12),
        color=ACCENT,
        arrowprops={"arrowstyle": "->", "color": ACCENT, "linewidth": 0.9},
    )
    figure.savefig(FIGURES / "convergence.png")
    plt.close(figure)


def golden_section_brackets() -> None:
    """What the search actually does: the bracket collapsing onto the optimum."""
    objective = Polynomial([-12.0, 8.0, -1.0])
    result = golden_section(objective, 0.0, 10.0, maximize=True, max_iterations=12)

    figure, axes = plt.subplots(figsize=(6.6, 3.4))
    grid = [i * 10.0 / 400 for i in range(401)]
    axes.plot(grid, [objective(x) for x in grid], color=MUTED, linewidth=1.2, zorder=1)

    for index, step in enumerate(result.history[:10]):
        axes.plot(
            [step.x],
            [objective(step.x)],
            "o",
            color=ACCENT,
            markersize=4.5,
            alpha=0.35 + 0.065 * index,
            zorder=3,
        )
        axes.annotate(str(index + 1), (step.x, objective(step.x)), textcoords="offset points",
                      xytext=(0, 7), ha="center", fontsize=6.5, color=ACCENT)

    axes.axvline(4.0, color=INK, linestyle=":", linewidth=1.0, zorder=2)
    axes.annotate(r"$x^\ast = 4$", xy=(4.0, -30), xytext=(4.4, -30), color=INK, fontsize=8)
    axes.set_xlabel("x")
    axes.set_ylabel("f(x)")
    axes.set_title(r"Golden-section probes on $-x^2 + 8x - 12$, first ten iterations", loc="left")
    figure.savefig(FIGURES / "golden-section.png")
    plt.close(figure)


def step_size_regimes() -> None:
    """The lesson of gradient descent: alpha decides everything."""
    objective, derivative = lambda x: (x - 3.0) ** 2, lambda x: 2.0 * (x - 3.0)
    regimes = [
        (0.02, "too small: crawls", PALETTE[2]),
        (0.4, "well chosen", PALETTE[3]),
        (0.95, "near the limit: oscillates", PALETTE[1]),
        (1.05, r"$\alpha > 2/L$: diverges", ACCENT),
    ]

    figure, axes = plt.subplots(figsize=(6.6, 3.6))
    for offset, (rate, label, colour) in enumerate(regimes):
        result = gradient_descent(
            objective, derivative, 0.0, learning_rate=rate, tolerance=0.0, max_iterations=40
        )
        errors = [max(abs(step.x - 3.0), 1e-17) for step in result.history]
        style = "--" if not result.converged and rate > 1.0 else "-"
        axes.semilogy(
            range(1, len(errors) + 1), errors, style, color=colour, linewidth=1.6,
            marker=MARKERS[offset], markersize=5, markerfacecolor="white",
            markeredgewidth=1.2, markevery=(offset * 2, 6), label=f"a={rate}: {label}",
        )

    axes.set_xlabel("iteration")
    axes.set_ylabel(r"$|x_k - x^\ast|$")
    axes.set_title(r"Gradient descent on $(x-3)^2$, where $L = 2$ so the limit is $\alpha = 1$", loc="left")
    axes.legend(loc="lower left", fontsize=7.5)
    figure.savefig(FIGURES / "step-size.png")
    plt.close(figure)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    convergence_rates()
    golden_section_brackets()
    step_size_regimes()
    print(f"wrote {len(list(FIGURES.glob('*.png')))} figures to {FIGURES}")
