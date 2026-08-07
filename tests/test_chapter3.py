"""Momentum, Adam, annealing and the Robbins-Monro conditions."""

import pytest

from khumbu import adam, momentum, robbins_monro, simulated_annealing


def bowl(v):
    return sum(vi * vi for vi in v)


def bowl_gradient(v):
    return [2 * vi for vi in v]


def narrow_valley(v):
    """Badly conditioned: 100x steeper in one direction than the other."""
    x, y = v
    return x * x + 100 * y * y


def narrow_valley_gradient(v):
    x, y = v
    return [2 * x, 200 * y]


def test_momentum_reaches_the_minimum() -> None:
    result = momentum(bowl, bowl_gradient, [3.0, -2.0], learning_rate=0.05, decay=0.9)
    assert result.converged
    assert result.fx == pytest.approx(0.0, abs=1e-6)


def test_momentum_beats_plain_descent_in_a_narrow_valley() -> None:
    """The zig-zag that momentum exists to cure, measured."""
    from khumbu.modern import momentum as run

    with_memory = run(
        narrow_valley,
        narrow_valley_gradient,
        [1.0, 1.0],
        learning_rate=0.005,
        decay=0.9,
        max_iterations=300,
    )
    without = run(
        narrow_valley,
        narrow_valley_gradient,
        [1.0, 1.0],
        learning_rate=0.005,
        decay=0.0,
        max_iterations=300,
    )
    assert with_memory.fx < without.fx


def test_nesterov_looks_ahead_and_still_converges() -> None:
    result = momentum(
        bowl, bowl_gradient, [3.0, -2.0], learning_rate=0.05, decay=0.9, nesterov=True
    )
    assert result.fx == pytest.approx(0.0, abs=1e-6)


def test_momentum_rejects_a_decay_that_would_never_forget() -> None:
    with pytest.raises(ValueError, match=r"decay must lie in \[0, 1\)"):
        momentum(bowl, bowl_gradient, [1.0], decay=1.0)


def test_adam_reaches_the_minimum() -> None:
    result = adam(bowl, bowl_gradient, [3.0, -2.0], learning_rate=0.1, max_iterations=2000)
    assert result.fx == pytest.approx(0.0, abs=1e-6)


def test_adam_bias_correction_matters_on_the_first_step() -> None:
    """Without the correction the first step would be about a tenth of its size.

    At iteration 1 with beta1 = 0.9 the raw average holds only (1 - beta1) of
    the true gradient. The correction divides by exactly that, so the first
    step has full length -- close to the learning rate itself.
    """
    result = adam(bowl, bowl_gradient, [1.0], learning_rate=0.1, max_iterations=1)
    moved = abs(1.0 - result.x[0])
    assert moved == pytest.approx(0.1, rel=0.05)


def test_adam_rejects_a_decay_outside_its_range() -> None:
    with pytest.raises(ValueError, match=r"beta2 must lie in \[0, 1\)"):
        adam(bowl, bowl_gradient, [1.0], beta2=1.0)


def test_annealing_escapes_a_local_minimum_that_traps_descent() -> None:
    """A double well: the deeper basin is at -2, the trap at +2."""

    def double_well(v):
        (x,) = v
        return (x * x - 4) ** 2 + 0.6 * x

    # The barrier between the wells is f(0) - f(2) = 14.8, so a schedule
    # starting at T = 1 accepts a crossing with probability exp(-14.8) ~ 4e-7:
    # effectively never. The initial temperature must be on the scale of the
    # barrier for annealing to do the one thing it exists for.
    result = simulated_annealing(
        double_well,
        [2.0],
        initial_temperature=20.0,
        final_temperature=1e-3,
        iterations=6000,
        step_size=0.8,
        seed=3,
    )
    assert result.x[0] < 0  # crossed the barrier into the deeper well


def test_annealing_is_reproducible_under_a_seed() -> None:
    first = simulated_annealing(bowl, [4.0], iterations=500, seed=11)
    second = simulated_annealing(bowl, [4.0], iterations=500, seed=11)
    assert first.x == second.x


def test_annealing_returns_the_best_point_not_the_last() -> None:
    result = simulated_annealing(bowl, [5.0], iterations=800, seed=1)
    assert result.fx == pytest.approx(min(result.history))


def test_annealing_requires_a_schedule_that_cools() -> None:
    with pytest.raises(ValueError, match="must cool"):
        simulated_annealing(bowl, [1.0], initial_temperature=0.1, final_temperature=1.0)


def test_robbins_monro_converges_through_noise() -> None:
    import random

    rng = random.Random(5)

    def noisy(v):
        return [2 * vi + rng.gauss(0.0, 0.5) for vi in v]

    result = robbins_monro(noisy, [5.0], scale=0.5, exponent=0.7, iterations=20_000)
    assert abs(result.x[0]) < 0.2


def test_robbins_monro_rejects_an_exponent_that_breaks_the_conditions() -> None:
    """P <= 0.5 makes the squared steps diverge; p > 1 makes travel finite."""
    with pytest.raises(ValueError, match=r"must lie in \(0.5, 1\]"):
        robbins_monro(lambda v: v, [1.0], exponent=0.5)
    with pytest.raises(ValueError, match=r"must lie in \(0.5, 1\]"):
        robbins_monro(lambda v: v, [1.0], exponent=1.5)
