from scripts.adaptive_beta_controller import score_population


def test_population_score_prefers_quality_and_isotropy() -> None:
    better = {
        seed: {
            "empirical_wasserstein_distance": 0.2,
            "pareto_step_empirical_wasserstein_distance": {"5": 0.22},
            "mean_eigenvalue_spread": 0.4,
            "missing_modes": 0,
        }
        for seed in range(3)
    }
    worse = {
        seed: {
            "empirical_wasserstein_distance": 0.3,
            "pareto_step_empirical_wasserstein_distance": {"5": 0.32},
            "mean_eigenvalue_spread": 0.6,
            "missing_modes": 0,
        }
        for seed in range(3)
    }

    assert score_population(better) < score_population(worse)
