from contextrag.eval.stats import (bootstrap_mean_ci, cliffs_delta_from_deltas,
                                   cohen_d_from_deltas,
                                   equivalence_and_noninferiority,
                                   holm_bonferroni_adjust, mean,
                                   paired_randomization_p_value,
                                   standard_deviation)


def test_mean():
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert mean([]) == 0.0


def test_bootstrap_mean_ci_deterministic_seed():
    low, high = bootstrap_mean_ci([0.0, 1.0, 2.0], seed=7, n_resamples=200)
    assert low <= high
    assert low <= 1.0 <= high


def test_paired_randomization_p_value():
    assert paired_randomization_p_value([0.0, 0.0, 0.0]) == 1.0
    p_value = paired_randomization_p_value([1.0, 1.0, 1.0], n_trials=1000, seed=7)
    assert 0.0 <= p_value <= 1.0


def test_standard_deviation():
    assert standard_deviation([]) == 0.0
    assert standard_deviation([1.0]) == 0.0
    assert round(standard_deviation([1.0, 2.0, 3.0]), 6) == 1.0


def test_effect_sizes_from_deltas():
    deltas = [0.2, 0.1, -0.1, 0.0]
    assert cohen_d_from_deltas(deltas) != 0.0
    assert cliffs_delta_from_deltas(deltas) == 0.25


def test_holm_bonferroni_adjust():
    adjusted = holm_bonferroni_adjust(
        {"a": 0.01, "b": 0.02, "c": 0.2}
    )
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    assert 0.0 <= adjusted["a"] <= 1.0


def test_equivalence_and_noninferiority():
    result = equivalence_and_noninferiority(ci_low=-0.01, ci_high=0.01, margin=0.02)
    assert result["equivalent_within_margin"] is True
    assert result["non_inferior_within_margin"] is True
    assert result["superior_to_zero"] is False
