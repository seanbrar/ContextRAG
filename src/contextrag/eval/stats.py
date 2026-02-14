"""Statistical helpers for paired evaluation comparisons."""

from __future__ import annotations

import math
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


def standard_deviation(values: list[float]) -> float:
    """Sample standard deviation."""
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def cohen_d_from_deltas(deltas: list[float]) -> float:
    """Paired-effect Cohen's d using per-query deltas."""
    if not deltas:
        return 0.0
    std = standard_deviation(deltas)
    if std == 0:
        return 0.0
    return mean(deltas) / std


def cliffs_delta_from_deltas(deltas: list[float]) -> float:
    """One-sample Cliff's delta against zero using sign dominance."""
    if not deltas:
        return 0.0
    positive = sum(1 for value in deltas if value > 0)
    negative = sum(1 for value in deltas if value < 0)
    return (positive - negative) / len(deltas)


def holm_bonferroni_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values by metric key."""
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted_pairs: list[tuple[str, float]] = []
    running_max = 0.0
    for idx, (metric, p_value) in enumerate(ordered, start=1):
        multiplier = m - idx + 1
        adjusted = min(1.0, p_value * multiplier)
        running_max = max(running_max, adjusted)
        adjusted_pairs.append((metric, running_max))
    return {metric: adjusted for metric, adjusted in adjusted_pairs}


def equivalence_and_noninferiority(
    ci_low: float,
    ci_high: float,
    margin: float,
) -> dict[str, bool]:
    """Decision helpers based on CI bounds and a symmetric margin."""
    return {
        "equivalent_within_margin": ci_low > -margin and ci_high < margin,
        "non_inferior_within_margin": ci_low > -margin,
        "superior_to_zero": ci_low > 0,
    }
