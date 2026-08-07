"""A reproducible benchmark, because a new optimiser is a claim.

The 2025 survey of pretraining optimisers found that *"many reported speedups
of 2x simply reflect a weak baseline"* (arXiv:2509.02046). This module exists so
that the claims made in this package's own README can be checked, and so that
anyone adding a method has to measure it against a baseline they cannot quietly
under-tune.

The protocol, stated before any result:

* **Fixed evaluation budget, not a fixed iteration count.** Methods whose steps
  cost differently -- SAM pays two gradients per step -- are charged for it.
* **Thirty seeds.** A single run of a stochastic method is an anecdote.
* **Median and interquartile range**, never the best run. Reporting the best
  seed is how a method is made to look better than it is.
* **Hyperparameters chosen on calibration seeds disjoint from the evaluation
  seeds**, so the reported number is not the one that was tuned on.

Run it with ``python -m khumbu.benchmark``.
"""

from khumbu.benchmark.functions import PROBLEMS, Problem
from khumbu.benchmark.runner import Outcome, run_benchmark

__all__ = ["PROBLEMS", "Outcome", "Problem", "run_benchmark"]
