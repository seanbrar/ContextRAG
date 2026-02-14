from contextrag.eval.stats import (bootstrap_mean_ci, mean,
                                   paired_randomization_p_value)


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
