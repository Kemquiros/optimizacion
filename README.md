# numopt — classical numerical optimization

[![CI](https://github.com/Kemquiros/optimizacion/actions/workflows/ci.yml/badge.svg)](https://github.com/Kemquiros/optimizacion/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Tested implementations of the classical one-dimensional optimization methods, written so that
each run can be **inspected rather than trusted**: every routine returns the full iterate
history, and a run that exhausts its budget without meeting its tolerance reports
`converged=False` instead of quietly returning its last point.

The algorithms were first written in 2017 for the undergraduate course *Optimización* at
Universidad de Antioquia. They are preserved verbatim in [`legacy/`](legacy/); this package is
a rewrite, and the differences are documented in [Provenance](#provenance).

---

## Install

```bash
pip install git+https://github.com/Kemquiros/optimizacion
```

No runtime dependencies. Python 3.11 or newer.

## Use

```python
from numopt import Polynomial, golden_section, newton_raphson, gradient_descent

# -x^2 + 8x - 12, written in ascending-degree order
f = Polynomial([-12.0, 8.0, -1.0])

result = golden_section(f, a=0.0, b=10.0, maximize=True)
print(result.x, result.converged, result.iterations)  # 4.0 True 42

# Newton on the derivative solves a quadratic in one step
result = newton_raphson(f.derivative(), f.derivative().derivative(), x0=0.0)
print(result.history[0].x, result.iterations)  # 4.0 2  (exact at step 1)

for step in result.history:
    print(step.iteration, step.x, step.error)
```

---

## Methods

| Function | Requires | Convergence | Guarantee |
|---|---|---|---|
| `golden_section(f, a, b)` | unimodality on `[a, b]` | linear, ratio `(√5−1)/2 ≈ 0.618` | bracket always shrinks |
| `bisection(df, a, b)` | sign change of `f′` | linear, ratio `1/2` | error bound `(b−a)/2ⁿ`, deterministic |
| `newton_raphson(df, d2f, x₀)` | `f″ ≠ 0` near the root | quadratic, locally | none globally — may diverge |
| `gradient_descent(f, df, x₀)` | `L`-Lipschitz `f′`, `α < 2/L` | linear | none if `α` too large |
| `stochastic_gradient_descent(...)` | as above | linear in expectation | escapes shallow minima; needs a seed to be checkable |
| `conjugate_gradient(A, b)` | `A` symmetric positive-definite | ≤ `n` steps exactly | terminates in exact arithmetic |

### Golden-section search

On a unimodal interval the method keeps two interior points at

$$x_1 = b - \rho\,(b-a), \qquad x_2 = a + \rho\,(b-a), \qquad \rho = \frac{\sqrt5 - 1}{2}$$

and discards the sub-interval that cannot contain the optimum. The value of $\rho$ is chosen so
that one of the two interior points is **reused** in the next iteration — that is the whole
point of the golden ratio here, and it halves the number of function evaluations relative to a
naive trisection.

### Bisection on the derivative

A stationary point satisfies $f'(x) = 0$. Given $f'(a)\,f'(b) < 0$ the bracket is halved each
step, so after $n$ iterations the error is at most $(b-a)/2^{\,n}$ — known *before* the run
starts. This is the only method here whose error bound does not depend on the function.

If the derivative does not change sign across the interval, the function raises rather than
returning an endpoint: bisection has no guarantee to offer there, and saying so is more useful
than a plausible wrong answer.

### Newton–Raphson

$$x_{k+1} = x_k - \frac{f'(x_k)}{f''(x_k)}$$

Quadratic convergence near a simple root, no guarantee away from one. A quadratic objective is
solved in a single step — the first iterate is already exact, and the test suite asserts it.

### Gradient descent

$$x_{k+1} = x_k - \alpha\,f'(x_k)$$

Convergence requires $\alpha < 2/L$ for an $L$-Lipschitz gradient. No line search is performed,
so the step size is the caller's responsibility — and when it is too large the result reports
`converged=False` rather than pretending.

The stochastic variant adds Gaussian noise annealed as $\sigma/\sqrt{k}$, which lets the iterate
leave a shallow basin while still settling.

### Conjugate gradient

Solves $A x = b$ for symmetric positive-definite $A$, equivalently minimizing
$\tfrac12 x^\top A x - b^\top x$. Search directions are $A$-conjugate, so in exact arithmetic the
method terminates in at most $n$ steps — the property that separates it from steepest descent,
which zig-zags on ill-conditioned quadratics.

---

## Development

```bash
git clone https://github.com/Kemquiros/optimizacion && cd optimizacion
pip install -e ".[dev]"

pytest          # test suite
ruff check .    # lint
mypy            # strict type checking
```

CI runs all three on Python 3.11, 3.12 and 3.13.

---

## Provenance

The 2017 scripts are kept unmodified under [`legacy/`](legacy/) as a record of the original
coursework. They are Python 2, interactive-only, and depend on `matplotlib` and the removed
`compiler` module. They are **not** importable and are not covered by CI.

What changed in the rewrite, and why:

| 2017 | Now | Reason |
|---|---|---|
| Python 2, `compiler.parse`, `raw_input` | Python 3.11+ | The `compiler` module was removed in Python 3.0; the scripts cannot run on any supported interpreter |
| `eval()` on the objective typed by the user | callables passed as arguments | Arbitrary code execution from input |
| Algorithms interleaved with `matplotlib` calls and `input()` prompts | pure functions, no I/O | The algorithms could not be called from other code, or tested at all |
| Stopping flag misspelled `termin` instead of `termina` in the deterministic gradient-descent script | fixed, with a regression test | **The loop never terminated.** The bug is preserved in `legacy/` and pinned by `test_gradient_descent_terminates` |
| Newton drew a fresh random start when `f″ = 0` | raises `ZeroDivisionError` | Retrying hides a real failure of the method |
| Unseeded `random` in the stochastic method | explicit `seed` argument | An unseeded stochastic result cannot be reproduced by a reader |
| No convergence signal | `converged: bool` on every result | A budget-exhausted run and a converged run were indistinguishable |

Original coursework: John Edisson Tapias Zarrazola, Universidad de Antioquia, 2017.

## Citation

See [`CITATION.cff`](CITATION.cff), or:

> Tapias Zarrazola, J. E. *numopt: classical one-dimensional numerical optimization methods.*
> Version 1.0.0, 2026. https://github.com/Kemquiros/optimizacion

## License

MIT — see [`LICENSE`](LICENSE).
