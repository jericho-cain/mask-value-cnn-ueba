"""
CNN Autoencoder for UEBA Behavioral Pattern Detection

This module implements a CNN-based autoencoder that operates on UEBA behavioral data
treated as 2D "images" where time and features are spatial coordinates.

Following the successful approach from gravitational wave detection, this model:
- Treats (time_steps, features) as spatial coordinates in "data-space-time"
- Learns latent representations of behavioral "shapes" 
- Enables manifold learning on these behavioral pattern embeddings

Key features:
- Input dimensions: (T, F) = (time buckets, behavioral features), T and F must be divisible by 4
- Spatial compression: T×F → (T/2)×(F/2) → (T/4)×(F/4) → latent_dim
- Decoder outputs exact dimensions (no interpolation)
- Linear output head for z-scored regression (no tanh clipping)
- Trains on normal behavioral patterns
- Detects anomalies through reconstruction error + manifold geometry

Architecture is flexible to handle different window sizes (12h, 24h, 48h, etc.).
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class UEBACNNAutoencoder(nn.Module):
    """
    CNN-based Autoencoder for UEBA behavioral pattern detection.
    
    Treats behavioral sequences as 2D images where time and features are spatial
    coordinates, enabling geometric analysis in the learned latent space.
    
    This follows the "data-space-time" concept: just as gravitational waves have
    (frequency, time, amplitude) coordinates, UEBA has (time, features, activity) 
    coordinates that form geometric patterns.
    
    Parameters
    ----------
    time_steps : int
        Number of time steps in behavioral sequences (typically 24 for 24 hours)
    n_features : int
        Number of behavioral features (12 for CERT after removing file_to_removable duplicate)
    latent_dim : int, optional
        Dimension of the latent space representation, by default 32
    dropout : float, optional
        Dropout rate for regularization, by default 0.1
        
    Attributes
    ----------
    time_steps : int
        Number of time steps in sequences
    n_features : int
        Number of behavioral features
    latent_dim : int
        Dimension of the latent space
    encoder : nn.Sequential
        CNN encoder for spatial feature extraction
    decoder : nn.Sequential
        CNN decoder for behavioral pattern reconstruction
        
    Examples
    --------
    >>> # CERT UEBA dimensions
    >>> model = UEBACNNAutoencoder(time_steps=24, n_features=12, latent_dim=32)
    >>> x = torch.randn(32, 1, 24, 12)  # Batch of CERT behavioral sequences
    >>> reconstructed, latent = model(x)
    >>> print(f"Reconstructed shape: {reconstructed.shape}")
    >>> print(f"Latent shape: {latent.shape}")
    """
    
    def __init__(
        self, 
        time_steps: int = 24,
        n_features: int = 12,
        latent_dim: int = 32,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        self.time_steps = time_steps
        self.n_features = n_features
        self.latent_dim = latent_dim
        
        # Compute spatial dimensions after convolutions
        # conv1: stride=1 → no change
        # conv2: stride=2 → T//2, F//2
        # conv3: stride=2 → T//4, F//4
        self.encoded_time = time_steps // 4
        self.encoded_features = n_features // 4
        self.flattened_size = 64 * self.encoded_time * self.encoded_features
        
        # Encoder: Extract behavioral pattern features from (time, features) space
        # Architecture designed specifically for 24×12 CERT dimensions
        # Spatial path: 24×12 → 12×6 → 6×3 → latent_dim
        self.encoder = nn.Sequential(
            # First conv: capture local time-feature correlations
            # 24×12 → 24×12
            # NOTE: in_channels=2 for Ablation C (mask + value dual-channel)
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            # Second conv: capture broader behavioral patterns
            # 24×12 → 12×6
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), 
            nn.Dropout2d(dropout),
            
            # Third conv: high-level behavioral abstractions
            # 12×6 → 6×3
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            # Flatten spatial dimensions: (T//4)×(F//4)×64
            nn.Flatten(),
            
            # Map to latent space
            nn.Linear(self.flattened_size, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Decoder: Reconstruct behavioral patterns from latent space
        # Architecture designed to output exactly T×F without interpolation
        # Spatial path: latent_dim → (T//4)×(F//4) → (T//2)×(F//2) → T×F
        self.decoder = nn.Sequential(
            # Map from latent back to spatial features
            nn.Linear(latent_dim, self.flattened_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Unflatten(1, (64, self.encoded_time, self.encoded_features)),
            
            # Upsampling convolutions to reconstruct behavioral patterns
            # 6×3 → 12×6 (output_padding=(1,1) for exact dimensions)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.ReLU(),
            
            # 12×6 → 24×12 (output_padding=(1,1) for exact dimensions)
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.ReLU(),
            
            # Final reconstruction layer: 24×12 → 24×12
            # NOTE: out_channels=2 for Ablation C: [mask_logits, value_hat]
            nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1)
            # Linear output: channel 0 = mask logits, channel 1 = value (no activation)
        )
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the UEBA CNN autoencoder.
        
        Encodes behavioral sequences to latent space and decodes back to 
        reconstructed behavioral patterns.
        
        Parameters
        ----------
        x : torch.Tensor
            Input behavioral sequence tensor of shape (batch_size, 1, time_steps, n_features)
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - reconstructed: Reconstructed behavioral sequence (batch_size, 1, time_steps, n_features)
            - latent: Latent behavioral pattern representation (batch_size, latent_dim)
        """
        # Encode to latent behavioral pattern representation
        latent = self.encoder(x)
        
        # Decode back to behavioral sequence (outputs exact dimensions: 24×13)
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode behavioral sequences to latent representations.
        
        Useful for manifold analysis where we only need latent embeddings.
        
        Parameters
        ----------
        x : torch.Tensor
            Input behavioral sequences, shape (batch_size, 1, time_steps, n_features)
            
        Returns
        -------
        torch.Tensor
            Latent representations, shape (batch_size, latent_dim)
        """
        return self.encoder(x)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations back to behavioral sequences.
        
        Parameters
        ----------
        latent : torch.Tensor
            Latent representations, shape (batch_size, latent_dim)
            
        Returns
        -------
        torch.Tensor
            Reconstructed behavioral sequences, shape (batch_size, 1, time_steps, n_features)
        """
        # Decoder outputs exact dimensions (24×13), no interpolation needed
        return self.decoder(latent)
    
    def get_model_info(self) -> dict[str, Any]:
        """
        Get model architecture information.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing model architecture details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'time_steps': self.time_steps,
            'n_features': self.n_features,
            'latent_dim': self.latent_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'architecture': 'UEBA CNN Autoencoder'
        }


def sequence_mse_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Per-sequence mean squared error for 2D CNN outputs.
    
    Computes MSE between original and reconstructed behavioral sequences,
    flattening the spatial dimensions (time_steps, n_features).
    
    Parameters
    ----------
    x, y : torch.Tensor
        Tensors of shape (batch_size, 1, time_steps, n_features)
        
    Returns
    -------
    torch.Tensor
        Tensor of shape (batch_size,) with one MSE score per sequence
    """
    return ((x - y) ** 2).mean(dim=(1, 2, 3))


def two_term_mse_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    raw_target: torch.Tensor,
    lambda_active: float = 2.0
) -> torch.Tensor:
    """
    Two-term MSE loss: MSE on all + bonus MSE on active buckets.
    
    Gentler alternative to weighted loss. Preserves baseline reconstruction
    while still encouraging learning of activity patterns.
    
    loss = MSE(all) + λ * MSE(active_only)
    
    Parameters
    ----------
    reconstructed : torch.Tensor
        Reconstructed sequences (z-scored), shape (batch, 1, time, features)
    target : torch.Tensor  
        Target sequences (z-scored), shape (batch, 1, time, features)
    raw_target : torch.Tensor
        Raw count data BEFORE z-scoring, shape (batch, time, features)
        Used only to compute activity mask
    lambda_active : float
        Weight for active-only term, default 2.0
        Try: {1, 2, 5}
        
    Returns
    -------
    torch.Tensor
        Scalar two-term MSE loss
    """
    # Standard MSE on all buckets (preserves baseline reconstruction)
    loss_all = torch.mean((reconstructed - target) ** 2)
    
    # Compute activity mask from raw counts
    active_mask = (raw_target > 0).float().unsqueeze(1)  # (batch, 1, time, features)
    
    # Additional loss on active buckets only
    # Only compute where mask=1, avoid dividing by zero
    num_active = torch.sum(active_mask)
    if num_active > 0:
        loss_active = torch.sum(active_mask * (reconstructed - target) ** 2) / num_active
    else:
        loss_active = torch.tensor(0.0, device=reconstructed.device)
    
    # Combined loss
    return loss_all + lambda_active * loss_active


def masked_value_mse(
    value_hat: torch.Tensor,
    value_true: torch.Tensor,
    mask_true: torch.Tensor
) -> torch.Tensor:
    """
    MSE for value reconstruction, computed only on active cells (mask_true==1).
    
    Normalized by number of active cells to avoid batch size / sparsity effects.
    
    Parameters
    ----------
    value_hat : torch.Tensor
        Reconstructed values (z-scored), shape (B, 1, T, F)
    value_true : torch.Tensor
        Target values (z-scored), shape (B, 1, T, F)
    mask_true : torch.Tensor
        Binary mask {0, 1}, shape (B, 1, T, F)
        
    Returns
    -------
    torch.Tensor
        Scalar MSE averaged over active cells only
    """
    # Squared error
    se = (value_hat - value_true) ** 2
    se = se * mask_true
    denom = mask_true.sum().clamp_min(1.0)
    return se.sum() / denom


class MaskValueLoss(nn.Module):
    """
    Dual-channel loss for mask + value reconstruction (Ablation C).
    
    loss = BCEWithLogits(mask_logits, mask_true) + λ_value * masked_MSE(value_hat, value_true | mask_true)
          + λ_temporal * ||z_t+1 - z_t||^2  (optional, for Block 2)
    
    The mask channel captures sparsity/density patterns (primary signal: ~75% of AE performance).
    The value channel captures magnitude patterns conditioned on presence (remaining ~25%).
    The temporal term encourages smooth latent trajectories for consecutive normal windows.
    
    Parameters
    ----------
    pos_weight : float
        Weight for positive class in BCE loss (to handle class imbalance).
        Typically: (# inactive cells) / (# active cells) ≈ 11-12 for CERT.
    lambda_value : float, optional
        Weight for masked value MSE term, default 1.0
    lambda_temporal : float, optional
        Weight for temporal smoothness penalty, default 0.0 (disabled)
        For Block 2: try 0.01-0.02
        
    Examples
    --------
    >>> # Compute pos_weight from training data
    >>> pos = (train_data > 0).sum()
    >>> neg = (train_data == 0).sum()
    >>> pos_weight = float(neg / pos)
    >>> 
    >>> criterion = MaskValueLoss(pos_weight=pos_weight, lambda_value=1.0, lambda_temporal=0.01)
    >>> recon, latent = model(x)  # recon: (B, 2, T, F), latent: (B, D)
    >>> losses = criterion(recon, mask_true, value_true, latent_t=latent, latent_t1=latent_next)
    >>> losses["loss"].backward()
    """
    
    def __init__(self, pos_weight: float, lambda_value: float = 1.0, lambda_temporal: float = 0.0) -> None:
        super().__init__()
        pw = torch.tensor([pos_weight], dtype=torch.float32)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.lambda_value = lambda_value
        self.lambda_temporal = lambda_temporal

    def forward(
        self,
        recon: torch.Tensor,      # (B, 2, T, F)
        mask_true: torch.Tensor,  # (B, 1, T, F)
        value_true: torch.Tensor, # (B, 1, T, F)
        latent_t: torch.Tensor = None,   # (B, D) optional for temporal reg
        latent_t1: torch.Tensor = None,  # (B, D) optional for temporal reg
        recon_t1: torch.Tensor = None,   # (B, 2, T, F) optional reconstruction at t+1
        mask_true_t1: torch.Tensor = None,   # (B, 1, T, F) optional
        value_true_t1: torch.Tensor = None,  # (B, 1, T, F) optional
    ) -> dict:
        """
        Compute dual-channel loss with optional temporal regularization.
        
        Parameters
        ----------
        recon : torch.Tensor
            Reconstructed output at time t, shape (B, 2, T, F)
            Channel 0: mask logits (no sigmoid)
            Channel 1: value estimates (linear)
        mask_true : torch.Tensor
            True binary mask at time t {0, 1}, shape (B, 1, T, F)
        value_true : torch.Tensor
            True z-scored values at time t, shape (B, 1, T, F)
        latent_t : torch.Tensor, optional
            Latent representation at time t, shape (B, D)
        latent_t1 : torch.Tensor, optional
            Latent representation at time t+1, shape (B, D)
        recon_t1 : torch.Tensor, optional
            Reconstructed output at time t+1 (for reconstruction loss), shape (B, 2, T, F)
        mask_true_t1 : torch.Tensor, optional
            True binary mask at time t+1, shape (B, 1, T, F)
        value_true_t1 : torch.Tensor, optional
            True z-scored values at time t+1, shape (B, 1, T, F)
            
        Returns
        -------
        dict
            Dictionary with keys:
            - "loss": total loss for backprop
            - "mask_loss": BCE component (detached, for logging)
            - "value_loss": masked MSE component (detached, for logging)
            - "temporal_loss": temporal smoothness component (detached, for logging, 0.0 if disabled)
        """
        # Reconstruction loss at time t
        mask_logits = recon[:, 0:1, :, :]  # (B, 1, T, F)
        value_hat = recon[:, 1:2, :, :]    # (B, 1, T, F)

        mask_loss_t = self.bce(mask_logits, mask_true)
        value_loss_t = masked_value_mse(value_hat, value_true, mask_true)

        # If we have t+1 data, add reconstruction loss for it too (for stability)
        if recon_t1 is not None and mask_true_t1 is not None and value_true_t1 is not None:
            mask_logits_t1 = recon_t1[:, 0:1, :, :]  # (B, 1, T, F)
            value_hat_t1 = recon_t1[:, 1:2, :, :]    # (B, 1, T, F)
            
            mask_loss_t1 = self.bce(mask_logits_t1, mask_true_t1)
            value_loss_t1 = masked_value_mse(value_hat_t1, value_true_t1, mask_true_t1)
            
            mask_loss = (mask_loss_t + mask_loss_t1) / 2.0
            value_loss = (value_loss_t + value_loss_t1) / 2.0
        else:
            mask_loss = mask_loss_t
            value_loss = value_loss_t

        total = mask_loss + (self.lambda_value * value_loss)
        
        # Add temporal regularization if enabled
        temporal_loss = torch.tensor(0.0, device=total.device, dtype=total.dtype)
        if self.lambda_temporal > 0 and latent_t is not None and latent_t1 is not None:
            # Penalize large jumps between consecutive latents
            temporal_loss = torch.mean((latent_t1 - latent_t) ** 2)
            total = total + (self.lambda_temporal * temporal_loss)
        
        return {
            "loss": total,
            "mask_loss": mask_loss.detach(),
            "value_loss": value_loss.detach(),
            "temporal_loss": temporal_loss.detach(),
        }


def weighted_mse_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    raw_target: torch.Tensor,
    w_inactive: float = 1.0,
    w_active: float = 20.0
) -> torch.Tensor:
    """
    Weighted MSE loss that emphasizes active (non-zero) values over sparse baseline.
    
    Problem: With z-scoring, zeros become negative constants that the model can
    fit extremely well. This causes the AE to optimize for the inactive baseline
    instead of actual activity spikes.
    
    Solution: Weight reconstruction errors by activity presence in raw space,
    making spikes matter more than the sparse background.
    
    Parameters
    ----------
    reconstructed : torch.Tensor
        Reconstructed sequences (z-scored), shape (batch, 1, time, features)
    target : torch.Tensor  
        Target sequences (z-scored), shape (batch, 1, time, features)
    raw_target : torch.Tensor
        Raw count data BEFORE z-scoring, shape (batch, time, features)
        Used only to compute activity mask
    w_inactive : float
        Weight for inactive buckets (raw count = 0), default 1.0
    w_active : float
        Weight for active buckets (raw count > 0), default 20.0
        
    Returns
    -------
    torch.Tensor
        Scalar weighted MSE loss
        
    Example
    -------
    >>> raw_counts = train_data  # (batch, 24, 13) before z-scoring
    >>> z_scored = batch  # (batch, 24, 13) after z-scoring
    >>> z_scored_4d = z_scored.unsqueeze(1)  # (batch, 1, 24, 13)
    >>> reconstructed, latent = model(z_scored_4d)
    >>> loss = weighted_mse_loss(reconstructed, z_scored_4d, raw_counts, w_active=20.0)
    """
    # Compute activity mask from raw counts (before z-scoring)
    # Shape: (batch, time, features)
    active_mask = (raw_target > 0).float()
    
    # Add channel dimension to match reconstructed/target
    # Shape: (batch, 1, time, features)
    active_mask = active_mask.unsqueeze(1)
    
    # Compute per-element weights
    weights = w_inactive + (w_active - w_inactive) * active_mask
    
    # Weighted squared error
    squared_error = (reconstructed - target) ** 2
    weighted_error = weights * squared_error
    
    # Return mean over all dimensions
    return weighted_error.mean()


def prepare_ueba_sequences_for_cnn(sequences: torch.Tensor) -> torch.Tensor:
    """
    Prepare UEBA sequences for CNN processing.
    
    Converts standard UEBA sequence format to CNN input format.
    Assumes sequences are already z-score normalized.
    
    Parameters
    ----------
    sequences : torch.Tensor
        UEBA sequences of shape (batch_size, time_steps, n_features) or
        (batch_size, 1, time_steps, n_features)
        Already z-score normalized via SeqDataset
        
    Returns
    -------
    torch.Tensor
        CNN-ready sequences of shape (batch_size, 1, time_steps, n_features)
    """
    # Handle different input shapes
    if sequences.dim() == 3:
        # Add channel dimension: (batch, time, features) -> (batch, 1, time, features)
        sequences = sequences.unsqueeze(1)
    elif sequences.dim() == 4 and sequences.shape[1] != 1:
        raise ValueError(f"Expected 1 channel, got {sequences.shape[1]}")
    
    return sequences.float()


def create_ueba_cnn_model(
    time_steps: int = 24,
    n_features: int = 12,
    latent_dim: int = 32,
    **kwargs
) -> UEBACNNAutoencoder:
    """
    Factory function to create UEBA CNN autoencoder model.
    
    Parameters
    ----------
    time_steps : int
        Number of time steps in behavioral sequences
    n_features : int
        Number of behavioral features
    latent_dim : int
        Latent space dimension
    **kwargs
        Additional arguments passed to UEBACNNAutoencoder
        
    Returns
    -------
    UEBACNNAutoencoder
        Initialized CNN autoencoder model
    """
    return UEBACNNAutoencoder(
        time_steps=time_steps,
        n_features=n_features,
        latent_dim=latent_dim,
        **kwargs
    )