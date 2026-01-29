"""
Scoring functions for window-level anomaly detection.

Provides scoring methods for different model types and loss configurations.
"""

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def score_windows_mask_value(
    model,
    windows_raw: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    device: torch.device,
) -> dict:
    """
    Score windows using mask + value dual-channel reconstruction.
    
    For Ablation C: Returns separate scores for mask and value reconstruction,
    plus a combined score.
    
    Parameters
    ----------
    model : UEBACNNAutoencoder
        Trained autoencoder with 2-channel output
    windows_raw : np.ndarray
        Raw count data, shape (N, T, F)
    mu : np.ndarray
        Feature means for z-scoring, shape (F,)
    sigma : np.ndarray
        Feature stds for z-scoring, shape (F,)
    device : torch.device
        Device to run computations on
        
    Returns
    -------
    dict
        Dictionary with keys:
        - "mask_bce": Per-window BCE for mask reconstruction (higher = more anomalous)
        - "value_mse_active": Per-window MSE on active cells only (higher = more anomalous)
        - "total": mask_bce + value_mse_active (simple sum; can tune weights)
        - "latent": Latent representations, shape (N, latent_dim)
    
    Examples
    --------
    >>> scores = score_windows_mask_value(model, test_data, mu, sigma, device)
    >>> ae_only_score = scores["total"]  # Combined anomaly score
    >>> pr_auc = average_precision_score(test_labels, ae_only_score)
    """
    model.eval()

    raw = torch.tensor(windows_raw, dtype=torch.float32, device=device)  # (N, T, F)

    mu_t = torch.tensor(mu, dtype=torch.float32, device=device).view(1, 1, -1)
    sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device).view(1, 1, -1)

    value = (raw - mu_t) / sigma_t  # (N, T, F) z-scored
    mask = (raw > 0).float()         # (N, T, F) binary {0, 1}

    x = torch.stack([mask, value], dim=1)  # (N, 2, T, F)

    recon, latent = model(x)  # recon: (N, 2, T, F)
    mask_logits = recon[:, 0:1, :, :]  # (N, 1, T, F)
    value_hat = recon[:, 1:2, :, :]    # (N, 1, T, F)

    mask_true = mask.unsqueeze(1)    # (N, 1, T, F)
    value_true = value.unsqueeze(1)  # (N, 1, T, F)

    # Per-element BCE (no reduction), then average over T, F
    bce = F.binary_cross_entropy_with_logits(mask_logits, mask_true, reduction="none")
    mask_bce = bce.mean(dim=(1, 2, 3)).cpu().numpy()

    # Active-only MSE
    se = (value_hat - value_true) ** 2
    se = se * mask_true
    denom = mask_true.sum(dim=(1, 2, 3)).clamp_min(1.0)  # Per-window active count
    value_mse_active = (se.sum(dim=(1, 2, 3)) / denom).cpu().numpy()

    total = mask_bce + value_mse_active

    return {
        "mask_bce": mask_bce,
        "value_mse_active": value_mse_active,
        "total": total,
        "latent": latent.detach().cpu().numpy(),
    }


@torch.no_grad()
def score_windows_standard(
    model,
    windows_zscore: np.ndarray,
    device: torch.device,
) -> dict:
    """
    Score windows using standard single-channel reconstruction (baseline).
    
    For exp007 baseline: Standard MSE reconstruction error.
    
    Parameters
    ----------
    model : UEBACNNAutoencoder
        Trained autoencoder with 1-channel output
    windows_zscore : np.ndarray
        Z-scored data, shape (N, T, F)
    device : torch.device
        Device to run computations on
        
    Returns
    -------
    dict
        Dictionary with keys:
        - "reconstruction_error": Per-window MSE (higher = more anomalous)
        - "latent": Latent representations, shape (N, latent_dim)
    """
    model.eval()

    x = torch.tensor(windows_zscore, dtype=torch.float32, device=device)  # (N, T, F)
    x = x.unsqueeze(1)  # (N, 1, T, F)

    recon, latent = model(x)

    # Per-window MSE
    mse = ((recon - x) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()

    return {
        "reconstruction_error": mse,
        "latent": latent.detach().cpu().numpy(),
    }
