"""
CNN Autoencoder for UEBA Behavioral Pattern Detection

This module implements a CNN-based autoencoder that operates on UEBA behavioral data
treated as 2D "images" where time and features are spatial coordinates.

Following the successful approach from gravitational wave detection, this model:
- Treats (time_steps, features) as spatial coordinates in "data-space-time"
- Learns latent representations of behavioral "shapes" 
- Enables manifold learning on these behavioral pattern embeddings

Key features:
- Input dimensions: (24, 13) = (time_steps, features)
- Trains on normal behavioral patterns
- Detects anomalies through reconstruction error + manifold geometry

Architecture is adapted from SimpleCWTAutoencoder but optimized for UEBA dimensions.
"""

from typing import Any

import torch
import torch.nn as nn


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
        Number of time steps in behavioral sequences (typically 24 for 6 hours)
    n_features : int
        Number of behavioral features (typically 13 for UEBA)
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
    >>> # UEBA dimensions
    >>> model = UEBACNNAutoencoder(time_steps=24, n_features=13, latent_dim=32)
    >>> x = torch.randn(32, 1, 24, 13)  # Batch of UEBA behavioral sequences
    >>> reconstructed, latent = model(x)
    >>> print(f"Reconstructed shape: {reconstructed.shape}")
    >>> print(f"Latent shape: {latent.shape}")
    """
    
    def __init__(
        self, 
        time_steps: int = 24,
        n_features: int = 13,
        latent_dim: int = 32,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        self.time_steps = time_steps
        self.n_features = n_features
        self.latent_dim = latent_dim
        
        # Encoder: Extract behavioral pattern features from (time, features) space
        # Note: UEBA dimensions (24, 13) are much smaller than GW (64, 3600)
        # so we use fewer layers and smaller kernels
        self.encoder = nn.Sequential(
            # First conv: capture local time-feature correlations
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            # Second conv: capture broader behavioral patterns
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(), 
            nn.Dropout2d(dropout),
            
            # Third conv: high-level behavioral abstractions
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            # Adaptive pooling to fixed size for variable input dimensions
            nn.AdaptiveAvgPool2d((4, 4)),  # Small fixed output for UEBA dimensions
            nn.Flatten(),
            
            # Map to latent space
            nn.Linear(64 * 4 * 4, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Decoder: Reconstruct behavioral patterns from latent space
        self.decoder = nn.Sequential(
            # Map from latent back to spatial features
            nn.Linear(latent_dim, 64 * 4 * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Unflatten(1, (64, 4, 4)),
            
            # Upsampling convolutions to reconstruct behavioral patterns
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # Final reconstruction layer
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Normalized output
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
            - reconstructed: Reconstructed behavioral sequence
            - latent: Latent behavioral pattern representation
        """
        # Encode to latent behavioral pattern representation
        latent = self.encoder(x)
        
        # Decode back to behavioral sequence
        reconstructed = self.decoder(latent)
        
        # Resize to original dimensions if needed (handle slight size differences)
        if reconstructed.shape[-2:] != (self.time_steps, self.n_features):
            reconstructed = torch.nn.functional.interpolate(
                reconstructed, 
                size=(self.time_steps, self.n_features), 
                mode='bilinear', 
                align_corners=False
            )
        
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
        reconstructed = self.decoder(latent)
        
        # Ensure correct output size
        if reconstructed.shape[-2:] != (self.time_steps, self.n_features):
            reconstructed = torch.nn.functional.interpolate(
                reconstructed,
                size=(self.time_steps, self.n_features),
                mode='bilinear',
                align_corners=False
            )
        
        return reconstructed
    
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


def prepare_ueba_sequences_for_cnn(sequences: torch.Tensor) -> torch.Tensor:
    """
    Prepare UEBA sequences for CNN processing.
    
    Converts standard UEBA sequence format to CNN input format and applies
    normalization suitable for spatial processing.
    
    Parameters
    ----------
    sequences : torch.Tensor
        UEBA sequences of shape (batch_size, time_steps, n_features) or
        (batch_size, 1, time_steps, n_features)
        
    Returns
    -------
    torch.Tensor
        CNN-ready sequences of shape (batch_size, 1, time_steps, n_features)
        with appropriate normalization
    """
    # Handle different input shapes
    if sequences.dim() == 3:
        # Add channel dimension: (batch, time, features) -> (batch, 1, time, features)
        sequences = sequences.unsqueeze(1)
    elif sequences.dim() == 4 and sequences.shape[1] != 1:
        raise ValueError(f"Expected 1 channel, got {sequences.shape[1]}")
    
    # Normalize for CNN processing (similar to image normalization)
    # This helps CNN learn spatial patterns in the behavioral data
    sequences = sequences.float()
    
    # Per-sample normalization to handle different user activity levels
    batch_size = sequences.shape[0]
    for i in range(batch_size):
        sample = sequences[i, 0]  # Remove channel dim for normalization
        
        # Robust normalization (handles zeros and outliers)
        sample_mean = torch.mean(sample)
        sample_std = torch.std(sample)
        
        if sample_std > 1e-8:  # Avoid division by zero
            sample = (sample - sample_mean) / sample_std
        
        # Scale to [-1, 1] range for tanh activation
        sample = torch.tanh(sample * 0.5)
        
        sequences[i, 0] = sample
    
    return sequences


def create_ueba_cnn_model(
    time_steps: int = 24,
    n_features: int = 13,
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