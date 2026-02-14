"""Statistical helpers for paired evaluation comparisons."""

from __future__ import annotations

import random


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def bootstrap_mean_ci(
    values: list[float],
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])

    alpha = 1.0 - confidence
    rng = random.Random(seed)
    n = len(values)
    samples: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(mean(sample))

    samples.sort()
    low_index = int((alpha / 2.0) * n_resamples)
    high_index = int((1.0 - alpha / 2.0) * n_resamples) - 1
    low_index = max(0, min(low_index, n_resamples - 1))
    high_index = max(0, min(high_index, n_resamples - 1))
    return (samples[low_index], samples[high_index])


def paired_randomization_p_value(
    deltas: list[float],
    n_trials: int = 5000,
    seed: int = 42,
) -> float:
    """Two-sided paired randomization test p-value for mean(delta) != 0."""
    if not deltas:
        return 1.0
    if all(delta == 0 for delta in deltas):
        return 1.0

    observed = abs(mean(deltas))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_trials):
        signed = [delta if rng.random() < 0.5 else -delta for delta in deltas]
        if abs(mean(signed)) >= observed:
            extreme += 1
    return (extreme + 1) / (n_trials + 1)
