"""Tests for window-level scoring (mask-value)."""

import numpy as np
import torch

import mv_ueba as pkg
from mv_ueba.scoring import score_windows_mask_value


def test_score_windows_mask_value_returns_expected_keys() -> None:
    """score_windows_mask_value returns dict with mask_bce, latent, and expected shapes.

    Returns
    -------
    None
        Asserts keys mask_bce, latent; shapes (N,) and (N, latent_dim); finite values.
    """
    N, T, F = 6, 24, 12
    latent_dim = 32
    model = pkg.UEBACNNAutoencoder(time_steps=T, n_features=F, latent_dim=latent_dim)
    model.eval()
    windows_raw = np.random.randint(0, 5, (N, T, F)).astype(np.float32)
    mu = np.zeros(F, dtype=np.float32)
    sigma = np.ones(F, dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    out = score_windows_mask_value(model, windows_raw, mu, sigma, device)

    assert "mask_bce" in out
    assert "latent" in out
    assert out["mask_bce"].shape == (N,)
    assert out["latent"].shape == (N, latent_dim)
    assert np.isfinite(out["mask_bce"]).all()
    assert np.isfinite(out["latent"]).all()


def test_score_windows_mask_value_has_total_or_ae_keys() -> None:
    """Scoring output includes combined/total score used by pipeline.

    Returns
    -------
    None
        Asserts 'total' or 'mask_bce' present; if total, shape (N,) and finite.
    """
    N, T, F = 4, 24, 12
    model = pkg.UEBACNNAutoencoder(time_steps=T, n_features=F, latent_dim=16)
    model.eval()
    windows_raw = np.random.randn(N, T, F).astype(np.float32)
    windows_raw = np.abs(windows_raw)  # non-negative for mask
    mu = windows_raw.reshape(-1, F).mean(axis=0)
    sigma = windows_raw.reshape(-1, F).std(axis=0) + 1e-10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    out = score_windows_mask_value(model, windows_raw, mu, sigma, device)

    # Pipeline uses total or similar for combined score
    assert "total" in out or "mask_bce" in out
    if "total" in out:
        assert out["total"].shape == (N,)
        assert np.isfinite(out["total"]).all()
