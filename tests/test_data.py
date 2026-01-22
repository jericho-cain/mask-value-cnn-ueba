import numpy as np

from manifold_ueba.data import (
    FEATURES,
    F,
    SeqDataset,
    compute_stats,
    make_sequence,
    make_timestep,
    standardize,
)


def test_features_and_F_consistency() -> None:
    assert isinstance(F, int) and F > 0
    assert isinstance(FEATURES, list) and len(FEATURES) == F
    assert len(set(FEATURES)) == len(FEATURES)


def test_make_timestep_shapes_and_types() -> None:
    for kind in ["normal", "spray_and_exfil", "geo_impossible_travel"]:
        x = make_timestep(kind)
        assert isinstance(x, np.ndarray)
        assert x.shape == (F,)
        assert x.dtype == np.float32
        assert np.isfinite(x).all()


def test_make_sequence_shapes() -> None:
    T = 10
    s1 = make_sequence(T=T, anomalous=False)
    s2 = make_sequence(T=T, anomalous=True, kind="spray_and_exfil")
    assert s1.shape == (T, F)
    assert s2.shape == (T, F)


def test_compute_stats_and_standardize_and_dataset() -> None:
    # Construct a simple dataset where per-feature mean is known
    base = np.arange(F, dtype=np.float32)
    train_array = np.tile(base, (5, 1))  # shape (5, F)
    mu, sigma = compute_stats(train_array)

    assert mu.shape == (F,)
    assert sigma.shape == (F,)
    assert np.allclose(mu, base, atol=1e-6)
    assert (sigma >= 1e-6).all()

    # Standardizing a sequence equal to mu should produce approx zeros
    seq = np.tile(mu, (3, 1))  # (T=3, F)
    z = standardize(seq, mu, sigma)
    assert z.shape == (3, F)
    assert np.allclose(z, 0.0, atol=1e-6)

    # Dataset wraps sequences and returns tensors
    ds = SeqDataset([seq], mu, sigma)
    assert len(ds) == 1
    item = ds[0]
    import torch

    assert isinstance(item, torch.Tensor)
    assert item.shape == (3, F)
