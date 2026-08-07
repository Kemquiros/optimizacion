"""AdamW, Lion, SAM and Muon."""

import pytest

from khumbu.frontier import _newton_schulz, adamw, lion, muon, sharpness_aware


def bowl(v):
    return sum(x * x for x in v)


def bowl_gradient(v):
    return [2 * x for x in v]


def test_adamw_reaches_the_minimum() -> None:
    result = adamw(
        bowl, bowl_gradient, [3.0, -2.0], learning_rate=0.1, weight_decay=0.0, max_iterations=3000
    )
    assert result.fx == pytest.approx(0.0, abs=1e-6)


def test_adamw_decay_is_decoupled_from_the_gradient() -> None:
    """The whole point of AdamW, isolated.

    With a zero gradient everywhere, Adam would not move at all. AdamW still
    shrinks the parameters, because the decay is applied to them directly
    rather than routed through the adaptive denominator.
    """
    flat = lambda v: 0.0  # noqa: E731
    zero_gradient = lambda v: [0.0 for _ in v]  # noqa: E731
    result = adamw(
        flat, zero_gradient, [1.0], learning_rate=0.1, weight_decay=0.5, max_iterations=10
    )
    assert abs(result.x[0]) < 1.0


def test_adamw_rejects_negative_weight_decay() -> None:
    with pytest.raises(ValueError, match="weight_decay must be non-negative"):
        adamw(bowl, bowl_gradient, [1.0], weight_decay=-0.1)


def test_lion_reaches_the_neighbourhood_of_the_minimum() -> None:
    result = lion(
        bowl, bowl_gradient, [1.0, -1.0], learning_rate=0.01, tolerance=0.0, max_iterations=500
    )
    assert result.fx < 0.01


def test_lion_moves_every_coordinate_by_exactly_the_learning_rate() -> None:
    """Check that step size is fully decoupled from gradient magnitude.

    That is the defining consequence of taking a sign: a coordinate with a huge
    gradient and one with a tiny gradient move exactly the same distance.
    """
    lopsided = lambda v: 1000 * v[0] ** 2 + 0.001 * v[1] ** 2  # noqa: E731
    lopsided_gradient = lambda v: [2000 * v[0], 0.002 * v[1]]  # noqa: E731
    start = [1.0, 1.0]
    result = lion(
        lopsided, lopsided_gradient, start, learning_rate=0.01, tolerance=0.0, max_iterations=1
    )
    moved = [abs(a - b) for a, b in zip(start, result.x, strict=True)]
    assert moved[0] == pytest.approx(moved[1], rel=1e-9)


def test_lion_rejects_a_coefficient_outside_its_range() -> None:
    with pytest.raises(ValueError, match=r"beta1 must lie in \[0, 1\)"):
        lion(bowl, bowl_gradient, [1.0], beta1=1.0)


def test_sam_charges_two_gradients_per_step() -> None:
    result = sharpness_aware(bowl, bowl_gradient, [2.0], max_iterations=10)
    assert result.evaluations == 2 * result.iterations


def test_sam_prefers_the_flat_minimum_of_two_equal_ones() -> None:
    """A sharp well at -1 and a flat one at +1, both with value 0.

    SAM evaluates the gradient after climbing to the worst point nearby, so the
    sharp well pushes it away harder than the flat one holds it.
    """

    def two_wells(v):
        (x,) = v
        return min(400.0 * (x + 1) ** 2, (x - 1) ** 2)

    def two_wells_gradient(v):
        (x,) = v
        return [800.0 * (x + 1)] if 400.0 * (x + 1) ** 2 < (x - 1) ** 2 else [2 * (x - 1)]

    result = sharpness_aware(
        two_wells, two_wells_gradient, [-0.9], learning_rate=0.02, radius=0.2, max_iterations=2000
    )
    assert result.x[0] > 0  # left the sharp well for the flat one


def test_sam_rejects_a_non_positive_radius() -> None:
    with pytest.raises(ValueError, match="radius must be positive"):
        sharpness_aware(bowl, bowl_gradient, [1.0], radius=0.0)


def _singular_values_2x2(m):
    """Singular values of a 2x2 matrix, via the eigenvalues of M^T M."""
    a = [[sum(m[k][i] * m[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    trace, det = a[0][0] + a[1][1], a[0][0] * a[1][1] - a[0][1] * a[1][0]
    disc = max(trace * trace / 4 - det, 0.0)
    return sorted((max(trace / 2 + disc**0.5, 0.0) ** 0.5, max(trace / 2 - disc**0.5, 0.0) ** 0.5))


def test_newton_schulz_compresses_the_spread_of_singular_values() -> None:
    """The guarantee is compression, not exact orthogonality.

    The quintic coefficients are tuned for speed inside a training loop, not for
    convergence: iterating further does not drive M^T M to the identity, it
    oscillates in a band. What the iteration *does* reliably do -- and what Muon
    needs -- is pull every singular value toward one, so no direction dominates
    the update. Measured here as the ratio between the largest and smallest.
    """
    matrix = [[8.0, 1.0], [1.0, 0.5]]
    before = _singular_values_2x2(matrix)
    after = _singular_values_2x2(_newton_schulz(matrix, steps=5))
    assert before[1] / before[0] > 20  # the input is badly spread
    assert after[1] / after[0] < 3  # the output is not
    for value in after:
        assert 0.5 < value < 1.5  # every singular value lands near one


def test_newton_schulz_does_not_converge_to_the_identity() -> None:
    """Documented as a property, so nobody 'fixes' it by iterating more."""
    matrix = [[3.0, 1.0], [1.0, 2.0]]
    five = _singular_values_2x2(_newton_schulz(matrix, steps=5))
    fifty = _singular_values_2x2(_newton_schulz(matrix, steps=50))
    assert abs(fifty[1] - 1.0) > 1e-3  # more steps do not mean more accuracy
    assert all(0.5 < v < 1.5 for v in five + fifty)


def test_muon_minimises_a_matrix_objective() -> None:
    def frobenius(m):
        return sum(v * v for row in m for v in row)

    def frobenius_gradient(m):
        return [[2 * v for v in row] for row in m]

    result = muon(
        frobenius,
        frobenius_gradient,
        [[1.0, 0.5], [0.5, 1.0]],
        learning_rate=0.05,
        max_iterations=200,
    )
    assert result.fx < frobenius([[1.0, 0.5], [0.5, 1.0]])


def test_muon_rejects_a_ragged_matrix() -> None:
    with pytest.raises(ValueError, match="must be rectangular"):
        muon(lambda m: 0.0, lambda m: m, [[1.0, 2.0], [3.0]])
