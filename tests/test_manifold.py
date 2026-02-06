"""Tests for latent manifold (k-NN, off-manifold distance)."""

import numpy as np

import mv_ueba as pkg


def test_manifold_fit_and_normal_deviation() -> None:
    """UEBALatentManifold fits on train latents; normal_deviation(z) expects single point (d,) and returns float.

    Returns
    -------
    None
        Asserts normal_deviation returns finite, non-negative float per test point.
    """
    np.random.seed(42)
    N_train, N_test, d = 50, 10, 32
    train_latents = np.random.randn(N_train, d).astype(np.float32) * 0.5
    test_latents = np.random.randn(N_test, d).astype(np.float32) * 0.5
    config = pkg.UEBAManifoldConfig(k_neighbors=5, tangent_dim=4)

    manifold = pkg.UEBALatentManifold(train_latents, config)
    for i in range(N_test):
        score = manifold.normal_deviation(test_latents[i])
        assert np.isfinite(score)
        assert score >= 0
        assert isinstance(score, (float, np.floating))


def test_manifold_density_score() -> None:
    """UEBALatentManifold.density_score returns a scalar float.

    Returns
    -------
    None
        Asserts score is float, finite, and non-negative.
    """
    np.random.seed(43)
    N, d = 30, 16
    train_latents = np.random.randn(N, d).astype(np.float32)
    config = pkg.UEBAManifoldConfig(k_neighbors=5)

    manifold = pkg.UEBALatentManifold(train_latents, config)
    one_point = np.random.randn(d).astype(np.float32)
    score = manifold.density_score(one_point)

    assert isinstance(score, (float, np.floating))
    assert np.isfinite(score)
    assert score >= 0
