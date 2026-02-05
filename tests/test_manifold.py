"""Tests for latent manifold (k-NN, off-manifold distance)."""

import numpy as np

import mv_ueba as pkg


def test_manifold_fit_and_normal_deviation() -> None:
    """UEBALatentManifold fits on train latents and returns finite normal_deviation."""
    np.random.seed(42)
    N_train, N_test, d = 50, 10, 32
    train_latents = np.random.randn(N_train, d).astype(np.float32) * 0.5
    test_latents = np.random.randn(N_test, d).astype(np.float32) * 0.5
    config = pkg.UEBAManifoldConfig(k_neighbors=5, tangent_dim=4)

    manifold = pkg.UEBALatentManifold(train_latents, config)
    scores = manifold.normal_deviation(test_latents)

    assert scores.shape == (N_test,)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


def test_manifold_density_score() -> None:
    """UEBALatentManifold.density_score returns a scalar float."""
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
