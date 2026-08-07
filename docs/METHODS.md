# The methods, one card each

A reference for the curious: for every algorithm in `khumbu`, the equation it runs, what it needs
in order to run, where it works, how fast, and **how it fails** — because the failure mode is what
tells you whether a method is right for your problem.

Read the [README](../README.md) first if you want the story. This is the lookup table.

**Notation.** `f` is the objective; `f′` and `f″` its first and second derivatives; `∇f` the
gradient and `H` the Hessian in several dimensions; `x*` the optimum; `n` the number of variables;
`k` the iteration index; `L` the Lipschitz constant of the gradient; `κ = λmax/λmin` the condition
number of the Hessian.

**How to read "convergence".** *Linear* with rate `r` means the error is multiplied by `r` each
step — one digit costs a fixed number of steps. *Superlinear* means `r` itself shrinks.
*Quadratic* means the number of correct digits doubles.

---

## Chapter 1 — Classical

### `golden_section(f, a, b)`

$$x_1 = b - \rho(b-a),\quad x_2 = a + \rho(b-a),\quad \rho = \tfrac{\sqrt5 - 1}{2} \approx 0.618$$

Discard whichever end cannot contain the optimum; repeat.

| | |
|---|---|
| **Domain** | one variable, on a bounded interval `[a, b]` |
| **Requires** | that `f` be **unimodal** on the interval — one optimum, no others. Nothing else: no derivative, no continuity of the derivative, not even continuity in principle |
| **Convergence** | linear, rate `0.618`. About **5 iterations per decimal digit** |
| **Cost per step** | one function evaluation (the other probe is reused — that is why the golden ratio appears) |
| **Guarantee** | the bracket shrinks monotonically and always contains the optimum |
| **Fails when** | the interval holds two minima. It converges confidently to one and never mentions the other. *Unimodality is an assumption you justify, not a checkbox* |
| **Use it when** | you cannot differentiate `f` and you can bracket the answer |

---

### `bisection(df, a, b)`

Find a sign change in the derivative: at an optimum, `f′(x*) = 0`.

$$\text{if } f'(a)\,f'(m) < 0 \text{ then } b \leftarrow m \text{ else } a \leftarrow m, \qquad m = \tfrac{a+b}{2}$$

| | |
|---|---|
| **Domain** | one variable, bounded interval |
| **Requires** | `f′`, and **a sign change** across the interval: `f′(a)·f′(b) < 0` |
| **Convergence** | linear, rate exactly `1/2` |
| **Error bound** | $(b-a)/2^{\,k}$ — **known before the run starts**, and independent of `f`. Ten digits over an interval of width 1 takes 34 iterations. Always |
| **Guarantee** | the strongest in this package: nothing about `f` can make it fail once bracketed |
| **Fails when** | there is no sign change. This implementation **raises** rather than returning an endpoint, because a plausible wrong answer costs more than an error |
| **Use it when** | you need a promise you can put in a proposal, or as the safe fallback when a fast method diverges |

---

### `brent(f, a, b)`

Fit a parabola through the three best points and jump to its vertex — *but only when that step is
provably sensible*: inside the bracket, and less than half the step before last. Otherwise take a
golden-section step.

| | |
|---|---|
| **Domain** | one variable, bounded interval |
| **Requires** | unimodality; no derivative |
| **Convergence** | superlinear on smooth functions; never worse than golden section |
| **Cost per step** | one function evaluation |
| **Guarantee** | golden section's floor, retained |
| **Fails when** | it does not, within its assumptions — which is the point. The hedge is the method |
| **Use it when** | you want a derivative-free minimiser and only get to choose one. This is what production libraries run |

---

### `newton_raphson(df, d2f, x₀)`

Model `f` locally as the parabola matching its value, slope and curvature; jump to that parabola's
minimum.

$$x_{k+1} = x_k - \frac{f'(x_k)}{f''(x_k)}$$

| | |
|---|---|
| **Domain** | one variable, unbounded |
| **Requires** | `f′` **and** `f″`, and `f″ ≠ 0` near the root |
| **Convergence** | **quadratic** near a simple root: correct digits roughly double each step. Three iterations can take you from 2 digits to 16 |
| **Cost per step** | one `f′` and one `f″` |
| **Guarantee** | **none globally.** Far from the optimum it can oscillate or diverge |
| **Fails when** | `f″ → 0`, where the step blows up. This implementation raises `ZeroDivisionError` instead of retrying from a random point, because retrying hides a genuine breakdown |
| **Use it when** | you have both derivatives and a starting guess you trust |

---

### `secant(df, x₀, x₁)`

Newton, with the second derivative estimated from the last two first derivatives.

$$x_{k+1} = x_k - f'(x_k)\,\frac{x_k - x_{k-1}}{f'(x_k) - f'(x_{k-1})}$$

| | |
|---|---|
| **Domain** | one variable, unbounded |
| **Requires** | `f′` only, and two distinct starting points |
| **Convergence** | superlinear with order **φ = (1+√5)/2 ≈ 1.618** — the golden ratio again, from a derivation entirely unrelated to golden-section search |
| **Cost per step** | **one** `f′`, versus Newton's two evaluations |
| **Guarantee** | none globally |
| **Fails when** | the two derivative values coincide: the secant line goes flat and the next iterate flies to infinity. Raises |
| **Use it when** | `f″` is expensive or unavailable. Slower per iteration than Newton, often **faster in total work** |

---

### `backtracking(f, df, x, d)`

Not a minimiser — a **step-length chooser**. Start optimistic, halve until the decrease is at least
proportional to what the slope promised (the Armijo condition):

$$f(x + t d) \le f(x) + c\,t\,\nabla f(x)^{\!\top} d, \qquad c \approx 10^{-4}$$

| | |
|---|---|
| **Domain** | any dimension |
| **Requires** | that `d` be a **descent direction**: `∇f(x)ᵀd < 0` |
| **Guarantee** | the returned step makes real progress, not merely downhill motion that shrinks faster than the step |
| **Fails when** | `d` points uphill. That is a caller bug and no step length rescues it, so it raises |
| **Use it when** | always, in place of a hand-tuned learning rate. It is what turns descent from a method that needs tuning into one that does not |

---

## Chapter 2 — Multivariate

### `nelder_mead(f, x₀)`

Keep `n+1` points — a simplex. Reflect the worst through the centroid of the others; expand if that
paid off, contract if it did not, shrink everything if nothing worked.

| | |
|---|---|
| **Domain** | `n` variables. Degrades badly above roughly **10 dimensions** |
| **Requires** | nothing but the ability to evaluate `f`. No gradient, no smoothness |
| **Convergence** | **no guarantee for `n > 1`** — a known theoretical result, not a defect of this implementation. It can stagnate on a degenerate simplex |
| **Cost per step** | 1–2 evaluations typically, `n+1` on a shrink |
| **Fails when** | dimensions grow, or the objective is noisy enough to reorder the vertices randomly |
| **Use it when** | `f` is a black box: a simulation, a physical experiment, legacy code nobody can differentiate |

---

### `bfgs(f, ∇f, x₀)`

Newton's speed without ever forming a Hessian. Accumulate an approximation `B ≈ H⁻¹` from gradients
already paid for, using the secant condition — curvature is visible in how the gradient changed:

$$B_{k+1} = \left(I - \rho s y^{\!\top}\right) B_k \left(I - \rho y s^{\!\top}\right) + \rho\, s s^{\!\top}, \qquad s = x_{k+1}-x_k,\; y = \nabla f_{k+1}-\nabla f_k,\; \rho = \tfrac{1}{y^{\!\top}s}$$

| | |
|---|---|
| **Domain** | `n` variables, unbounded. Practical to a few thousand dimensions; beyond that use L-BFGS, which stores the last `m` pairs instead of the full matrix |
| **Requires** | `∇f`. No Hessian, ever |
| **Convergence** | superlinear |
| **Cost per step** | one gradient, a line search, and `O(n²)` for the matrix update |
| **Memory** | `O(n²)` — the reason L-BFGS exists |
| **Guarantee** | descent is preserved: the curvature update is **skipped** when `yᵀs ≤ 0` would destroy positive-definiteness |
| **Fails when** | `n` is large enough that `n²` memory is unaffordable, or gradients are noisy — the secant condition assumes they are exact |
| **Use it when** | you have an exact gradient and fewer than a few thousand parameters. **This is the default that most optimisers actually run** |

---

### `conjugate_gradient(A, b)`

Solve `Ax = b` for symmetric positive-definite `A`, equivalently minimise `½xᵀAx − bᵀx`. Search
directions are `A`-conjugate: `dᵢᵀA dⱼ = 0`, so progress along one is never undone by the next.

| | |
|---|---|
| **Domain** | `n` variables, quadratic objective only |
| **Requires** | `A` symmetric **positive definite** |
| **Convergence** | **terminates in at most `n` steps** in exact arithmetic — a direct method wearing an iterative method's clothes. In floating point, error `≈ 2((√κ−1)/(√κ+1))ᵏ` |
| **Cost per step** | one matrix-vector product |
| **Memory** | `O(n)` — never forms `A⁻¹` |
| **Fails when** | `A` is indefinite. Raises when the curvature along a search direction is non-positive, because the alternative is a silently meaningless answer |
| **Use it when** | your problem *is* a large sparse SPD system, which is more often than people expect: least squares, finite elements, Gaussian processes |

---

## Chapter 3 — Modern

### `gradient_descent(f, f′, x₀)` and `stochastic_gradient_descent`

$$x_{k+1} = x_k - \alpha \nabla f(x_k)$$

| | |
|---|---|
| **Domain** | any dimension; the method that survives into millions |
| **Requires** | `∇f`, and a gradient that is `L`-Lipschitz for the bound below to hold |
| **Convergence** | linear with rate `(κ−1)/(κ+1)` on a quadratic. **Ill-conditioning is fatal**: at `κ = 1000` progress is roughly `0.998` per step |
| **Step size** | needs `α < 2/L`. Too small crawls; **above the bound the iterates diverge**, they do not merely slow |
| **Fails when** | `α` is misjudged, or the problem is ill-conditioned — see the benchmark, where plain descent scores `10³` on the 1000:1 bowl that BFGS solves to `10⁻²³` |
| **Use it when** | the dimension rules out anything storing curvature. Add `backtracking` and most of the difficulty disappears |

The stochastic variant adds noise annealed as `σ/√k`, which lets the iterate leave a shallow basin
while still settling. It takes a `seed`: an unseeded stochastic result cannot be checked by a reader.

---

### `momentum(f, ∇f, x₀, decay=β)`

$$v_{k+1} = \beta v_k - \alpha \nabla f(x_k), \qquad x_{k+1} = x_k + v_{k+1}$$

With `nesterov=True` the gradient is evaluated at the **look-ahead** point `x + βv` instead of at
`x`, so the method brakes before overshooting rather than after.

| | |
|---|---|
| **Domain** | any dimension |
| **Requires** | `∇f`; `β ∈ [0, 1)` — at `β = 1` the velocity never decays and the iterate never settles |
| **Convergence** | rate improves from `(κ−1)/(κ+1)` to roughly `(√κ−1)/(√κ+1)`: **the square root of the condition number**, which is the whole reason momentum exists |
| **Cost per step** | one gradient, one extra vector of memory |
| **Fails when** | `β` is too close to 1 for the curvature, producing sustained oscillation |
| **Use it when** | plain descent zig-zags — which is to say, nearly always |

---

### `adam(f, ∇f, x₀)`

$$m_k = \beta_1 m_{k-1} + (1-\beta_1)g_k, \qquad v_k = \beta_2 v_{k-1} + (1-\beta_2)g_k^2$$

$$\hat m = \frac{m_k}{1-\beta_1^k}, \quad \hat v = \frac{v_k}{1-\beta_2^k}, \qquad x_{k+1} = x_k - \alpha\,\frac{\hat m}{\sqrt{\hat v} + \varepsilon}$$

| | |
|---|---|
| **Domain** | any dimension; built for millions of parameters and noisy gradients |
| **Requires** | `∇f` (usually a mini-batch estimate); `β₁, β₂ ∈ [0, 1)` |
| **Convergence** | no clean guarantee on non-convex problems — the original convergence proof was **later shown to be flawed** (Reddi et al., 2018). It is used because it works, not because it is proven |
| **Cost per step** | one gradient, **two** extra vectors of memory |
| **Why the bias correction** | both averages start at zero, so at `k = 1` the first moment holds only `(1−β₁)` of the true gradient — about a tenth. Dividing by `1−β₁ᵏ` undoes exactly that shrinkage; without it the opening steps are far too small |
| **Fails when** | the problem is smooth, deterministic and low-dimensional. Then a quasi-Newton method beats it by orders of magnitude — see the benchmark |
| **Use it when** | gradients are noisy and per-coordinate scaling matters |

---

### `simulated_annealing(f, x₀)`

Accept a worse candidate with probability

$$P = \exp\!\left(-\frac{\Delta f}{T}\right), \qquad T_{k+1} = \alpha T_k, \quad \alpha = \left(\frac{T_\text{final}}{T_\text{initial}}\right)^{1/(N-1)}$$

The cooling rate is **derived from the schedule**, not guessed — the same construction used in
SIROA (2018).

| | |
|---|---|
| **Domain** | any dimension; continuous or combinatorial |
| **Requires** | only the ability to evaluate `f` and to perturb a candidate |
| **Convergence** | converges to the global optimum in probability **only under a logarithmic schedule** `T ∝ 1/log k`, which is far too slow to use. Every practical schedule, including this one, gives up that guarantee |
| **The critical parameter** | `T_initial` must be **on the scale of the barriers you need to cross**. A barrier of height 15 with `T = 1` is crossed with probability `e⁻¹⁵ ≈ 3×10⁻⁷`: never |
| **Fails when** | the schedule is too cold to escape, or so hot it never settles |
| **Use it when** | the landscape has many local minima. **It is the only method here that can leave one** — on Rastrigin it scores `0.059` where every gradient method scores `40.79` |

---

### `robbins_monro(noisy_gradient, x₀)`

Find a root of a function observable only through noise, with step sizes `aₖ = c/kᵖ`:

$$\sum_k a_k = \infty \quad \text{(can still travel any distance)} \qquad \sum_k a_k^2 < \infty \quad \text{(the noise averages out)}$$

Both hold exactly when `0.5 < p ≤ 1`.

| | |
|---|---|
| **Domain** | any dimension |
| **Requires** | an **unbiased** noisy estimate of the gradient |
| **Convergence** | almost sure, but **asymptotic**: any finite run is still moving, which is why this function reports no `converged` flag |
| **Why it matters** | those two conditions are the reason learning-rate decay is not a heuristic — and they are the same conditions under which **temporal-difference learning converges in reinforcement learning**. SGD and TD are the same theorem |
| **Fails when** | `p ≤ 0.5` (the squared steps diverge, so noise never averages out) or `p > 1` (total travel is finite, so a distant optimum is unreachable). Both raise |

---

## Chapter 4 — Frontier

### `adamw(f, ∇f, x₀, weight_decay=λ)`

$$x_{k+1} = x_k - \alpha\left(\frac{\hat m}{\sqrt{\hat v}+\varepsilon} + \lambda x_k\right)$$

| | |
|---|---|
| **The one change** | classical L2 adds `λx` to the *gradient*, and Adam then divides it by `√v̂` along with everything else — so **the effective regularisation depends on each coordinate's gradient history**. A rarely-updated parameter is decayed far more than a busy one. AdamW applies `λx` to the parameter directly, so it is uniform |
| **Convergence** | as Adam |
| **Fails when** | as Adam |
| **Use it when** | you would have used Adam and the model is regularised. **This is the baseline every newer method has to beat** — and, per the 2025 literature, mostly does not |

---

### `lion(f, ∇f, x₀)`

$$u_k = \operatorname{sign}\!\big(\beta_1 m_{k-1} + (1-\beta_1) g_k\big), \qquad x_{k+1} = x_k - \alpha(u_k + \lambda x_k)$$
$$m_k = \beta_2 m_{k-1} + (1-\beta_2) g_k$$

Note the asymmetry: `β₁` weights the **update**, `β₂` the **state**. Not the roles they play in Adam.

| | |
|---|---|
| **Origin** | discovered by symbolic program search (Chen et al., 2023), not derived |
| **Memory** | **half of Adam's** — one buffer, not two. At the scale where optimiser state competes with the model for memory, that is the entire argument |
| **The consequence of the sign** | every coordinate moves by exactly `α`, regardless of gradient magnitude. Lion is therefore insensitive to gradient scale and **very** sensitive to `α`: published recipes use roughly a tenth of Adam's |
| **Fails when** | the objective is smooth and low-dimensional, where a fixed step cannot reach machine precision — in the benchmark Lion plateaus around `10⁻⁴` where Adam reaches `10⁻¹⁸` |
| **Use it when** | memory is the binding constraint at large scale |

---

### `sharpness_aware(f, ∇f, x₀, radius=ρ)` — SAM

Minimise the **worst value in a neighbourhood**, not the value at the point:

$$\epsilon = \rho\,\frac{\nabla f(x)}{\lVert \nabla f(x)\rVert}, \qquad x_{k+1} = x_k - \alpha\,\nabla f(x_k + \epsilon)$$

| | |
|---|---|
| **The idea** | two minima with the same value are not equally good. A sharp one sits in a narrow crevasse, so a small shift in the data moves the loss a lot; a flat one tolerates it |
| **Cost per step** | **two gradients** — the price of asking a harder question. The benchmark charges it by halving the step budget |
| **Convergence** | to a flat region, not to the lowest value. On a deterministic benchmark it therefore *looks* worse, and is not being measured on what it optimises |
| **Fails when** | there is no generalisation gap to close — which is exactly the case in this package's benchmark |
| **Use it when** | validation accuracy matters more than training loss |

---

### `muon(f, ∇f, X₀)`

Treats a parameter as a **matrix**. Replace the momentum by its nearest orthogonal matrix, so no
singular direction dominates:

$$M_k = \mu M_{k-1} + G_k, \qquad O_k = \mathrm{NewtonSchulz}_5(M_k), \qquad X_{k+1} = X_k - \alpha O_k$$

with the quintic iteration `X ← aX + bX(XᵀX) + cX(XᵀX)²`, coefficients `(3.4445, −4.7750, 2.0315)`.

| | |
|---|---|
| **Domain** | matrix-shaped parameters — the weight matrices of a network |
| **Requires** | a rectangular gradient of the same shape |
| **The orthogonalisation does *not* converge** | and that is deliberate. The coefficients land singular values in a band around one — roughly `[0.7, 1.3]` — as fast as possible. Iterating further makes them **oscillate** inside that band rather than sharpen. Muon needs no direction to dominate, not exactness; an accurate orthogonalisation would cost the speedup and buy nothing |
| **Measured worth** | ~1.3× fewer steps below 520M parameters, **decaying to ~1.1× at 1.2B**, at **1.45× the wall time per step** ([arXiv:2509.02046](https://arxiv.org/abs/2509.02046)) |
| **Use it when** | you have measured it at your own scale. Whether the trade pays depends on where you sit on that curve |

---

## Choosing, in one page

```
Can you evaluate a gradient?
│
├─ No ──► one variable, bracketed? ──► brent  (or golden_section if you want the simplest)
│         many variables?           ──► nelder_mead   (below ~10 dimensions)
│         many local minima?        ──► simulated_annealing
│
└─ Yes
   ├─ Is the objective an SPD quadratic?      ──► conjugate_gradient
   ├─ Exact gradient, n < a few thousand?     ──► bfgs        ← usually the right answer
   ├─ Second derivative available and cheap?  ──► newton_raphson  (verify it converged)
   ├─ Noisy gradients, n in the millions?     ──► adamw
   │   ├─ memory-bound?                       ──► lion
   │   ├─ generalisation matters more?        ──► sharpness_aware
   │   └─ matrix parameters, measured gain?   ──► muon
   └─ Only a step size in doubt?              ──► backtracking, always
```

**The single most common mistake** is reaching for Chapter 3 or 4 on a Chapter 2 problem. Adam is
excellent on noisy million-parameter objectives and is beaten by fifteen orders of magnitude by
BFGS on a smooth two-variable one. The benchmark in this repository demonstrates exactly that, and
also states why it would be dishonest to conclude Adam is bad.
