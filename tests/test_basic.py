"""Basic package and model tests for mask-value CNN UEBA."""

import numpy as np
import torch

import mv_ueba as pkg


def test_version_present() -> None:
    """Package exposes a non-empty string __version__.

    Returns
    -------
    None
        Asserts pkg.__version__ is a non-empty string.
    """
    assert isinstance(pkg.__version__, str)
    assert pkg.__version__


def test_cnn_ae_forward_dual_channel() -> None:
    """CNN autoencoder expects 2-channel input (mask, value); forward and latent shapes.

    Returns
    -------
    None
        Asserts recon (B, 2, T, F), latent (B, 32), and sequence_mse_2d output (B,).
    """
    B, T, F = 4, 24, 12  # batch, time_steps, features (CERT uses 12)
    model = pkg.UEBACNNAutoencoder(time_steps=T, n_features=F, latent_dim=32)
    # Input: (B, 2, T, F) for mask + value channels
    x = torch.randn(B, 2, T, F)
    recon, z = model(x)
    assert recon.shape == (B, 2, T, F)
    assert z.shape == (B, 32)
    # Loss helper
    mse = pkg.sequence_mse_2d(recon, x)
    assert mse.shape == (B,)


def test_compute_stats() -> None:
    """compute_stats returns mu, sigma of shape (F,) from flattened (N*T, F).

    Returns
    -------
    None
        Asserts mu/sigma shape (F,), finite, and sigma non-negative.
    """
    N, T, F = 10, 24, 12
    data_flat = np.random.randn(N * T, F).astype(np.float32)
    mu, sigma = pkg.compute_stats(data_flat)
    assert mu.shape == (F,)
    assert sigma.shape == (F,)
    assert np.isfinite(mu).all() and np.isfinite(sigma).all()
    assert (sigma >= 0).all()


def test_mask_value_dataset_smoke() -> None:
    """MaskValueSeqDataset returns 2-channel (mask, value) and correct shapes.

    Returns
    -------
    None
        Asserts len, x/mask/value shapes, mask in [0,1], value finite.
    """
    N, T, F = 5, 24, 12
    raw = np.random.randint(0, 10, (N, T, F)).astype(np.float32)
    mu = raw.reshape(-1, F).mean(axis=0)
    sigma = raw.reshape(-1, F).std(axis=0) + 1e-10
    ds = pkg.MaskValueSeqDataset(raw, mu, sigma)
    assert len(ds) == N
    x, mask, value = ds[0]
    assert x.shape == (2, T, F)
    assert mask.shape == (1, T, F)
    assert value.shape == (1, T, F)
    assert mask.numpy().min() >= 0 and mask.numpy().max() <= 1  # binary mask
    assert torch.isfinite(value).all()


def test_seq_dataset_smoke() -> None:
    """SeqDataset returns z-scored sequence of shape (T, F).

    Returns
    -------
    None
        Asserts len, single-item shape (T, F), and finite values.
    """
    N, T, F = 5, 24, 12
    raw = np.random.randn(N, T, F).astype(np.float32)
    mu = raw.reshape(-1, F).mean(axis=0)
    sigma = raw.reshape(-1, F).std(axis=0) + 1e-10
    ds = pkg.SeqDataset(raw, mu, sigma)
    assert len(ds) == N
    z = ds[0]
    assert z.shape == (T, F)
    assert torch.isfinite(z).all()
