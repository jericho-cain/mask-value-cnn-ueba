"""Tests for CNN model and loss functions (mv_ueba.cnn_model).

All tests are smoke or shape/finite checks. Add parametrized edge cases
in new test functions; see tests/COVERAGE_MAP.md.
"""

import numpy as np
import pytest
import torch

import mv_ueba as pkg
from mv_ueba.cnn_model import (
    two_term_mse_loss,
    masked_value_mse,
    weighted_mse_loss,
    MaskValueLoss,
    prepare_ueba_sequences_for_cnn,
    create_ueba_cnn_model,
)


def test_two_term_mse_loss_shape_and_finite() -> None:
    """two_term_mse_loss returns scalar tensor, finite; handles all-inactive mask.

    Returns
    -------
    None
        Asserts loss is scalar, finite, and non-negative.
    """
    B, T, F = 4, 24, 12
    recon = torch.randn(B, 1, T, F).float()
    target = torch.randn(B, 1, T, F).float()
    raw = torch.zeros(B, T, F).float()  # all inactive
    loss = two_term_mse_loss(recon, target, raw, lambda_active=2.0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_two_term_mse_loss_with_active_mask() -> None:
    """two_term_mse_loss with some active cells returns finite scalar.

    Returns
    -------
    None
        Asserts loss is scalar and finite.
    """
    B, T, F = 2, 8, 4
    recon = torch.randn(B, 1, T, F).float()
    target = torch.randn(B, 1, T, F).float()
    raw = torch.randn(B, T, F).float().abs().clamp(0, 10)  # some active
    loss = two_term_mse_loss(recon, target, raw, lambda_active=1.0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_masked_value_mse_shape_and_finite() -> None:
    """masked_value_mse returns scalar; only active cells contribute.

    Returns
    -------
    None
        Asserts output is scalar and finite.
    """
    B, T, F = 4, 24, 12
    value_hat = torch.randn(B, 1, T, F).float()
    value_true = torch.randn(B, 1, T, F).float()
    mask_true = (torch.rand(B, 1, T, F) > 0.7).float()
    loss = masked_value_mse(value_hat, value_true, mask_true)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_masked_value_mse_all_zeros_mask() -> None:
    """masked_value_mse with all-zero mask does not raise; returns finite.

    Returns
    -------
    None
        Asserts no exception and finite scalar (denom handled in main code).
    """
    B, T, F = 2, 4, 4
    value_hat = torch.randn(B, 1, T, F).float()
    value_true = torch.randn(B, 1, T, F).float()
    mask_true = torch.zeros(B, 1, T, F).float()
    loss = masked_value_mse(value_hat, value_true, mask_true)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_mask_value_loss_forward_returns_keys() -> None:
    """MaskValueLoss forward returns dict with loss, mask_loss, value_loss, temporal_loss.

    Returns
    -------
    None
        Asserts keys present and loss scalar.
    """
    B, T, F, D = 4, 24, 12, 32
    criterion = MaskValueLoss(pos_weight=10.0, lambda_value=0.02, lambda_temporal=0.0)
    recon = torch.randn(B, 2, T, F).float()
    mask_true = (torch.rand(B, 1, T, F) > 0.5).float()
    value_true = torch.randn(B, 1, T, F).float()
    out = criterion(recon, mask_true, value_true)
    assert "loss" in out
    assert "mask_loss" in out
    assert "value_loss" in out
    assert "temporal_loss" in out
    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"])


def test_mask_value_loss_with_temporal() -> None:
    """MaskValueLoss with latent_t and latent_t1 adds temporal term.

    Returns
    -------
    None
        Asserts temporal_loss in output and total loss finite.
    """
    B, T, F, D = 2, 8, 4, 16
    criterion = MaskValueLoss(pos_weight=5.0, lambda_value=0.02, lambda_temporal=0.02)
    recon = torch.randn(B, 2, T, F).float()
    mask_true = (torch.rand(B, 1, T, F) > 0.5).float()
    value_true = torch.randn(B, 1, T, F).float()
    latent_t = torch.randn(B, D).float()
    latent_t1 = torch.randn(B, D).float()
    out = criterion(recon, mask_true, value_true, latent_t=latent_t, latent_t1=latent_t1)
    assert out["temporal_loss"].dim() == 0
    assert torch.isfinite(out["temporal_loss"])
    assert out["temporal_loss"].item() >= 0


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_weighted_mse_loss_shape(batch_size: int) -> None:
    """weighted_mse_loss returns scalar for different batch sizes.

    Parameters
    ----------
    batch_size : int
        Batch dimension for tensors.

    Returns
    -------
    None
        Asserts scalar, finite loss.
    """
    B, T, F = batch_size, 24, 12
    recon = torch.randn(B, 1, T, F).float()
    target = torch.randn(B, 1, T, F).float()
    raw = (torch.rand(B, T, F).float() > 0.5).float() * torch.rand(B, T, F).float()
    loss = weighted_mse_loss(recon, target, raw, w_inactive=1.0, w_active=20.0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_prepare_ueba_sequences_3d_adds_channel() -> None:
    """prepare_ueba_sequences_for_cnn (B,T,F) -> (B,1,T,F).

    Returns
    -------
    None
        Asserts output shape (B, 1, T, F).
    """
    B, T, F = 6, 24, 12
    seq = torch.randn(B, T, F).float()
    out = prepare_ueba_sequences_for_cnn(seq)
    assert out.shape == (B, 1, T, F)
    assert out.dtype == torch.float32


def test_prepare_ueba_sequences_4d_single_channel_passthrough() -> None:
    """prepare_ueba_sequences_for_cnn (B,1,T,F) unchanged.

    Returns
    -------
    None
        Asserts shape unchanged.
    """
    B, T, F = 4, 24, 12
    seq = torch.randn(B, 1, T, F).float()
    out = prepare_ueba_sequences_for_cnn(seq)
    assert out.shape == (B, 1, T, F)


def test_create_ueba_cnn_model_smoke() -> None:
    """create_ueba_cnn_model returns UEBACNNAutoencoder; default is 2-channel (mask+value).

    Returns
    -------
    None
        Asserts model forward and output shapes.
    """
    model = create_ueba_cnn_model(time_steps=24, n_features=12, latent_dim=32)
    assert model is not None
    x = torch.randn(2, 2, 24, 12).float()  # 2 channels: mask, value
    recon, z = model(x)
    assert recon.shape == (2, 2, 24, 12)
    assert z.shape == (2, 32)
