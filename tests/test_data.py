"""Tests for dataset classes (mv_ueba.data).

Smoke and shape checks. Add edge cases (e.g. single sample, all zeros)
as new parametrized tests; see tests/COVERAGE_MAP.md.
"""

import numpy as np
import pandas as pd
import pytest
import torch

import mv_ueba as pkg


def test_weighted_seq_dataset_len_and_getitem() -> None:
    """WeightedSeqDataset returns (z_scored, raw) per index; lengths match.

    Returns
    -------
    None
        Asserts __len__ and __getitem__ shapes.
    """
    N, T, F = 10, 24, 12
    raw = np.random.randn(N, T, F).astype(np.float32)
    mu = raw.reshape(-1, F).mean(axis=0)
    sigma = raw.reshape(-1, F).std(axis=0) + 1e-10
    ds = pkg.WeightedSeqDataset(raw, mu, sigma)
    assert len(ds) == N
    z, r = ds[0]
    assert z.shape == (T, F)
    assert r.shape == (T, F)
    assert torch.isfinite(z).all()
    assert torch.isfinite(r).all()


@pytest.mark.parametrize("n_samples", [1, 5, 20])
def test_weighted_seq_dataset_sizes(n_samples: int) -> None:
    """WeightedSeqDataset works for small and larger N.

    Parameters
    ----------
    n_samples : int
        Number of sequences in the dataset.

    Returns
    -------
    None
        Asserts len and one __getitem__.
    """
    N, T, F = n_samples, 24, 12
    raw = np.random.randn(N, T, F).astype(np.float32)
    mu = np.zeros(F, dtype=np.float32)
    sigma = np.ones(F, dtype=np.float32)
    ds = pkg.WeightedSeqDataset(raw, mu, sigma)
    assert len(ds) == N
    z, r = ds[0]
    assert z.shape == (T, F)


def test_temporal_paired_mask_value_dataset_smoke() -> None:
    """TemporalPairedMaskValueSeqDataset builds pairs from metadata; returns 6 tensors.

    Uses minimal metadata (two users, a few windows each) to avoid data dependency.
    Captures stdout from constructor (print) via capsys if needed; here we only assert shapes.

    Returns
    -------
    None
        Asserts __len__, __getitem__ returns 6 tensors with expected shapes.
    """
    N, T, F = 8, 24, 12
    np.random.seed(42)
    raw = np.random.randint(0, 5, (N, T, F)).astype(np.float32)
    mu = raw.reshape(-1, F).mean(axis=0)
    sigma = raw.reshape(-1, F).std(axis=0) + 1e-10
    # Two users: first 4 windows user A, next 4 user B; sort by window_start
    metadata = pd.DataFrame({
        "user_id": ["A"] * 4 + ["B"] * 4,
        "window_start": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"] * 2),
    })
    ds = pkg.TemporalPairedMaskValueSeqDataset(raw, metadata, mu, sigma)
    assert len(ds) >= 1  # At least one consecutive pair per user
    out = ds[0]
    assert len(out) == 6
    x_t, mask_t, value_t, x_t1, mask_t1, value_t1 = out
    assert x_t.shape == (2, T, F)
    assert mask_t.shape == (1, T, F)
    assert value_t.shape == (1, T, F)
    assert x_t1.shape == (2, T, F)
    assert mask_t1.shape == (1, T, F)
    assert value_t1.shape == (1, T, F)
    assert torch.isfinite(value_t).all() and torch.isfinite(value_t1).all()
