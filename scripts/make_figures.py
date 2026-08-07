"""Regenerate every figure in the README, by running the library.

    python scripts/make_figures.py

Nothing here is drawn by hand: a change in behaviour shows up as a changed
picture. Every series carries its own marker shape, so coincident curves stay
tellable apart, the figures survive black-and-white printing, and a colour-blind
reader loses nothing.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from khumbu import (  # noqa: E402
    Polynomial,
    adam,
    bfgs,
    bisection,
    brent,
    golden_section,
    gradient_descent,
    momentum,
    nelder_mead,
    newton_raphson,
    secant,
    simulated_annealing,
)
from khumbu.benchmark import PROBLEMS, run_benchmark  # noqa: E402
from khumbu.frontier import adamw, lion, sharpness_aware  # noqa: E402

FIGURES = Path(__file__).resolve().parent.parent / "figures"
INK, ACCENT = "#1a1a1a", "#c1440e"
PALETTE = ["#1a1a1a", "#c1440e", "#2166ac", "#4d9221", "#7b3294", "#b8860b"]
MARKERS = ["o", "s", "D", "^", "v", "P"]

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.family": "serif", "font.size": 9,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": "#dddddd", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def _plot_series(axes, series, xlabel, ylabel, title, log=True):
    """Draw named error curves with staggered markers."""
    for offset, (label, values) in enumerate(series.items()):
        plot = axes.semilogy if log else axes.plot
        plot(range(1, len(values) + 1), values, label=label,
             color=PALETTE[offset % len(PALETTE)], linewidth=1.6,
             marker=MARKERS[offset % len(MARKERS)], markersize=4.5,
             markerfacecolor="white", markeredgewidth=1.1,
             markevery=(offset * 2, max(len(values) // 8, 1)))
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title, loc="left")
    axes.legend(loc="upper right", fontsize=7.5)


def chapter1_convergence() -> None:
    """All six classical methods on one minimisation objective."""
    objective = Polynomial([16.0, -8.0, 1.0])  # (x - 4)^2
    d, dd = objective.derivative(), objective.derivative().derivative()
    runs = {
        "golden section": golden_section(objective, 0.0, 10.0, tolerance=1e-14, max_iterations=60),
        "brent": brent(objective, 0.0, 10.0, tolerance=1e-14, max_iterations=60),
        "bisection on f'": bisection(d, 0.0, 10.0, tolerance=1e-14, max_iterations=60),
        "secant": secant(d, 0.0, 1.0, tolerance=1e-14, max_iterations=60),
        "newton-raphson": newton_raphson(d, dd, 0.0, tolerance=1e-14, max_iterations=60),
        "gradient descent": gradient_descent(objective, d, 0.0, learning_rate=0.3,
                                             tolerance=1e-14, max_iterations=60),
    }
    series = {k: [max(abs(s.x - 4.0), 1e-17) for s in r.history] for k, r in runs.items()}
    figure, axes = plt.subplots(figsize=(7.0, 4.1))
    _plot_series(axes, series, "iteration", r"$|x_k - x^\ast|$",
                 "Chapter 1 — distance to the optimum, same objective and interval")
    axes.set_xlim(1, 45)
    axes.set_ylim(1e-17, 1e1)
    figure.savefig(FIGURES / "ch1-convergence.png")
    plt.close(figure)


def chapter1_step_size() -> None:
    """The four regimes of a fixed step size."""
    f, df = lambda x: (x - 3.0) ** 2, lambda x: 2.0 * (x - 3.0)  # noqa: E731
    regimes = [(0.02, "too small: crawls"), (0.4, "well chosen"),
               (0.95, "near the limit"), (1.05, r"$\alpha > 2/L$: diverges")]
    series = {}
    for rate, label in regimes:
        run = gradient_descent(f, df, 0.0, learning_rate=rate, tolerance=0.0, max_iterations=40)
        series[f"a={rate}: {label}"] = [max(abs(s.x - 3.0), 1e-17) for s in run.history]
    figure, axes = plt.subplots(figsize=(7.0, 3.8))
    _plot_series(axes, series, "iteration", r"$|x_k - x^\ast|$",
                 r"Chapter 1 — $(x-3)^2$ has $L = 2$, so the limit is $\alpha = 1$")
    figure.savefig(FIGURES / "ch1-step-size.png")
    plt.close(figure)


def chapter2_multivariate() -> None:
    """Nelder-Mead against BFGS on Rosenbrock, by iteration and by evaluation."""
    def rosen(v):
        x, y = v
        return (1 - x) ** 2 + 100 * (y - x * x) ** 2

    def rosen_grad(v):
        x, y = v
        return [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)]

    nm = nelder_mead(rosen, [-1.2, 1.0], tolerance=1e-12, max_iterations=800)
    qn = bfgs(rosen, rosen_grad, [-1.2, 1.0], tolerance=1e-12, max_iterations=200)

    figure, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for offset, (label, run) in enumerate({"nelder-mead": nm, "bfgs": qn}.items()):
        values = [max(v, 1e-17) for v in run.history]
        style = dict(color=PALETTE[offset], linewidth=1.6, marker=MARKERS[offset],
                     markersize=4.5, markerfacecolor="white", markeredgewidth=1.1,
                     markevery=(offset * 2, max(len(values) // 8, 1)), label=label)
        left.semilogy(range(1, len(values) + 1), values, **style)
        cost = run.evaluations / len(values)
        right.semilogy([cost * (i + 1) for i in range(len(values))], values, **style)
    left.set_xlabel("iteration"); right.set_xlabel("objective evaluations")
    for ax in (left, right):
        ax.set_ylabel("f(x)"); ax.legend(loc="upper right", fontsize=8)
    left.set_title("Chapter 2 — Rosenbrock, by iteration", loc="left")
    right.set_title("...and by what it actually cost", loc="left")
    figure.savefig(FIGURES / "ch2-multivariate.png")
    plt.close(figure)


def chapter3_modern() -> None:
    """The modern optimisers on a badly conditioned bowl."""
    def valley(v):
        return v[0] ** 2 + 100 * v[1] ** 2

    def valley_grad(v):
        return [2 * v[0], 200 * v[1]]

    start = [1.0, 1.0]
    runs = {
        "gradient descent": momentum(valley, valley_grad, start, learning_rate=0.004,
                                     decay=0.0, max_iterations=200),
        "momentum": momentum(valley, valley_grad, start, learning_rate=0.004,
                             decay=0.9, max_iterations=200),
        "nesterov": momentum(valley, valley_grad, start, learning_rate=0.004,
                             decay=0.9, nesterov=True, max_iterations=200),
        "adam": adam(valley, valley_grad, start, learning_rate=0.05, max_iterations=200),
    }
    series = {k: [max(v, 1e-17) for v in r.history] for k, r in runs.items()}
    figure, axes = plt.subplots(figsize=(7.0, 4.0))
    _plot_series(axes, series, "iteration", "f(x)",
                 "Chapter 3 — a 100:1 valley, where plain descent zig-zags")
    figure.savefig(FIGURES / "ch3-modern.png")
    plt.close(figure)


def chapter3_escaping() -> None:
    """The one thing gradients cannot do."""
    def rastrigin(v):
        return 10 + v[0] ** 2 - 10 * math.cos(2 * math.pi * v[0])

    def rastrigin_grad(v):
        return [2 * v[0] + 20 * math.pi * math.sin(2 * math.pi * v[0])]

    grid = [(-5.0 + 10.0 * i / 600) for i in range(601)]
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    left.plot(grid, [rastrigin([x]) for x in grid], color="#8a8a8a", linewidth=1.2)

    start = [4.4]
    descent = adam(rastrigin, rastrigin_grad, start, learning_rate=0.05, max_iterations=400)
    annealed = simulated_annealing(rastrigin, start, initial_temperature=12.0,
                                   final_temperature=1e-3, iterations=4000,
                                   step_size=0.7, seed=4)
    for offset, (label, x) in enumerate({"adam": descent.x[0], "annealing": annealed.x[0]}.items()):
        left.plot([x], [rastrigin([x])], MARKERS[offset], color=PALETTE[offset],
                  markersize=9, markerfacecolor="white", markeredgewidth=2, label=f"{label} ends here")
    left.plot([start[0]], [rastrigin(start)], "*", color=INK, markersize=13, label="start")
    left.set_xlabel("x"); left.set_ylabel("f(x)"); left.legend(loc="upper center", fontsize=7.5)
    left.set_title("Chapter 3 — Rastrigin: a lattice of traps", loc="left")

    series = {"adam": [max(v, 1e-12) for v in descent.history],
              "annealing": [max(v, 1e-12) for v in annealed.history]}
    _plot_series(right, series, "iteration", "best f(x) so far",
                 "Only the method that accepts worse steps escapes")
    figure.savefig(FIGURES / "ch3-escaping.png")
    plt.close(figure)


def chapter4_frontier() -> None:
    """Adam against everything published to replace it."""
    def valley(v):
        return v[0] ** 2 + 50 * v[1] ** 2

    def valley_grad(v):
        return [2 * v[0], 100 * v[1]]

    start = [1.0, 1.0]
    runs = {
        "adam": adam(valley, valley_grad, start, learning_rate=0.05, max_iterations=300),
        "adamw": adamw(valley, valley_grad, start, learning_rate=0.05,
                       weight_decay=0.01, max_iterations=300),
        "lion": lion(valley, valley_grad, start, learning_rate=0.01,
                     tolerance=0.0, max_iterations=300),
        "sam": sharpness_aware(valley, valley_grad, start, learning_rate=0.01,
                               max_iterations=300),
    }
    series = {k: [max(v, 1e-17) for v in r.history] for k, r in runs.items()}
    figure, axes = plt.subplots(figsize=(7.0, 4.0))
    _plot_series(axes, series, "iteration", "f(x)",
                 "Chapter 4 — the successors, on a problem none of them was designed for")
    axes.annotate("all four are within an order of magnitude;\nthe differences live at scale, not here",
                  xy=(0.36, 0.12), xycoords="axes fraction", fontsize=7.5, color=ACCENT)
    figure.savefig(FIGURES / "ch4-frontier.png")
    plt.close(figure)


def benchmark_summary() -> None:
    """The benchmark, drawn: median final value per method and problem."""
    outcomes = run_benchmark()
    problems = [p.name for p in PROBLEMS]
    methods = sorted({o.method for o in outcomes})
    lookup = {(o.problem, o.method): o.median for o in outcomes}

    figure, axes = plt.subplots(figsize=(9.6, 4.4))
    width = 0.8 / len(methods)
    for offset, method in enumerate(methods):
        values = [max(lookup.get((p, method), float("nan")), 1e-25) for p in problems]
        axes.bar([i + offset * width for i in range(len(problems))], values, width,
                 label=method, color=PALETTE[offset % len(PALETTE)],
                 alpha=0.85, edgecolor="white", linewidth=0.5)
    axes.set_yscale("log")
    axes.set_xticks([i + 0.4 - width / 2 for i in range(len(problems))])
    axes.set_xticklabels(problems)
    axes.set_ylabel("median final f(x) over 30 seeds (lower is better)")
    axes.set_title("The benchmark — equal evaluation budget, medians not best runs", loc="left")
    axes.legend(loc="upper left", fontsize=7, ncol=5)
    figure.savefig(FIGURES / "benchmark.png")
    plt.close(figure)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    for name, build in [
        ("ch1-convergence", chapter1_convergence), ("ch1-step-size", chapter1_step_size),
        ("ch2-multivariate", chapter2_multivariate), ("ch3-modern", chapter3_modern),
        ("ch3-escaping", chapter3_escaping), ("ch4-frontier", chapter4_frontier),
        ("benchmark", benchmark_summary),
    ]:
        build()
        print(f"  wrote {name}.png")
    print(f"{len(list(FIGURES.glob('*.png')))} figures in {FIGURES}")
