"""Smoke tests for trajectory analyzer (geodesic deviation)."""

import numpy as np

import manifold_ueba as pkg


def test_trajectory_analyzer_scores_sequence() -> None:
    """TrajectoryAnalyzer accepts a manifold and scores a trajectory (latent sequence)."""
    np.random.seed(44)
    N_train, d = 40, 32
    train_latents = np.random.randn(N_train, d).astype(np.float32) * 0.3
    config_manifold = pkg.UEBAManifoldConfig(k_neighbors=5)
    manifold = pkg.UEBALatentManifold(train_latents, config_manifold)

    traj_config = pkg.TrajectoryConfig()
    analyzer = pkg.TrajectoryAnalyzer(manifold, traj_config)

    # Short trajectory: 6 latent vectors (e.g. 6 days)
    trajectory = np.random.randn(6, d).astype(np.float32) * 0.2
    score = analyzer.score_trajectory(trajectory)

    assert isinstance(score, (float, np.floating)) or (isinstance(score, np.ndarray) and score.size == 1)
    assert np.isfinite(np.asarray(score)).all()
