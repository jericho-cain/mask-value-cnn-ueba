import manifold_ueba as pkg


def test_version_present() -> None:
    assert isinstance(pkg.__version__, str)
    assert pkg.__version__


def test_cnn_ae_forward_and_loss() -> None:
    """Test CNN autoencoder forward pass and loss computation."""
    B, T, F = 4, 24, 13  # batch, time_steps, features
    model = pkg.UEBACNNAutoencoder(time_steps=T, n_features=F, latent_dim=32)
    import torch

    # Create input in CNN format (B, 1, T, F)
    x = torch.randn(B, 1, T, F)
    recon, z = model(x)
    assert recon.shape == (B, 1, T, F)
    assert z.shape == (B, 32)
    mse = pkg.sequence_mse_2d(recon, x)
    assert mse.shape == (B,)
