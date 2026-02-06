"""Smoke tests for trajectory analyzer and trajectory construction."""

import numpy as np

import mv_ueba as pkg
from mv_ueba.trajectory import create_trajectories_from_sequences


def test_create_trajectories_from_sequences_groups_by_user() -> None:
    """create_trajectories_from_sequences returns dict user_id -> list of (seq_len, d) arrays.

    Returns
    -------
    None
        Asserts dict keys, list of arrays per user, each trajectory shape (seq_len, d), finite.
    """
    np.random.seed(45)
    d = 8
    seq_len = 6
    # 2 users: user 0 has 10 latents, user 1 has 10 latents
    latent_sequences = np.random.randn(20, d).astype(np.float32)
    user_ids = np.array([0] * 10 + [1] * 10)
    out = create_trajectories_from_sequences(latent_sequences, user_ids, sequence_length=seq_len)
    assert isinstance(out, dict)
    assert 0 in out and 1 in out
    for uid in (0, 1):
        trajs = out[uid]
        assert isinstance(trajs, list)
        for t in trajs:
            assert t.shape == (seq_len, d)
            assert np.isfinite(t).all()
    # Each user has 10 points, non-overlapping chunks of 6 -> 1 trajectory per user
    assert len(out[0]) >= 1
    assert len(out[1]) >= 1


def test_trajectory_analyzer_scores_sequence() -> None:
    """TrajectoryAnalyzer accepts a manifold and scores a trajectory (latent sequence).

    Returns
    -------
    None
        Asserts score is scalar (float or size-1 array) and finite.
    """
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
