"""
Manifold-Based Scoring for UEBA Behavioral Pattern Detection

This module combines CNN autoencoder reconstruction error with manifold geometry
to detect anomalous behavioral patterns in UEBA data.

Scoring formula:
    combined_score = alpha * reconstruction_error + beta * off_manifold_distance

The beta coefficient directly measures whether manifold geometry helps detect
behavioral anomalies that reconstruction error alone cannot capture!

For UEBA:
- alpha weight for CNN reconstruction (how well behavioral pattern reconstructs)
- beta weight for manifold geometry (how far behavioral pattern is off-manifold)  
- Grid search over alpha, beta to find optimal combination

Adapted from successful gravitational wave detection approach.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from .cnn_model import sequence_mse_2d
from .latent_manifold import UEBALatentManifold

ScoringMode = Literal["cnn_only", "manifold_only", "cnn_plus_manifold"]


@dataclass
class UEBAManifoldScorerConfig:
    """
    Configuration for UEBA manifold-based scoring.
    
    Parameters
    ----------
    mode : ScoringMode, optional
        Scoring mode ('cnn_only', 'manifold_only', 'cnn_plus_manifold'), 
        by default 'cnn_plus_manifold'
    alpha_cnn : float, optional
        Weight for CNN reconstruction error, by default 1.0
    beta_manifold : float, optional
        Weight for off-manifold distance (normal deviation), by default 1.0
    use_density : bool, optional
        Whether to include density term, by default False
    gamma_density : float, optional
        Weight for density score (if used), by default 0.0
        
    Examples
    --------
    >>> # CNN only (baseline)
    >>> config = UEBAManifoldScorerConfig(mode='cnn_only')
    >>> 
    >>> # CNN + manifold (the key test for β > 0)
    >>> config = UEBAManifoldScorerConfig(
    ...     mode='cnn_plus_manifold',
    ...     alpha_cnn=1.0,
    ...     beta_manifold=2.0
    ... )
    """
    mode: ScoringMode = "cnn_plus_manifold"
    alpha_cnn: float = 1.0          # weight for CNN reconstruction error
    beta_manifold: float = 1.0      # weight for normal deviation (β component!)
    use_density: bool = False       # optional second manifold term
    gamma_density: float = 0.0      # weight for density (if used)


class UEBAManifoldScorer:
    """
    Combines CNN reconstruction error with manifold geometry for UEBA anomaly detection.
    
    For UEBA, this combines:
    - Reconstruction error: How well does CNN reconstruct the behavioral pattern?
    - Off-manifold distance: How far is latent from normal behavioral manifold?
    
    The beta coefficient measures the importance of manifold geometry for detecting
    behavioral anomalies. β > 0 indicates manifold learning provides value beyond
    reconstruction error alone.
    
    Parameters
    ----------
    manifold : UEBALatentManifold
        Built manifold from normal behavioral pattern latents
    config : UEBAManifoldScorerConfig
        Scoring configuration (alpha, beta weights)
        
    Attributes
    ----------
    manifold : UEBALatentManifold
        The manifold instance built from normal behavioral patterns
    config : UEBAManifoldScorerConfig
        Scoring configuration
        
    Examples
    --------
    >>> # Build manifold from normal behavioral pattern training data
    >>> manifold = UEBALatentManifold(train_latents, manifold_config)
    >>> 
    >>> # Create scorer with alpha=1.0, beta=2.0
    >>> config = UEBAManifoldScorerConfig(alpha_cnn=1.0, beta_manifold=2.0)
    >>> scorer = UEBAManifoldScorer(manifold, config)
    >>> 
    >>> # Score behavioral sequences
    >>> scores = scorer.score_batch(cnn_model, behavioral_sequences)
    """

    def __init__(self, manifold: UEBALatentManifold, config: UEBAManifoldScorerConfig):
        """
        Initialize UEBA manifold scorer.
        
        Parameters
        ----------
        manifold : UEBALatentManifold
            Built manifold from normal behavioral patterns
        config : UEBAManifoldScorerConfig
            Scoring configuration
        """
        self.manifold = manifold
        self.config = config

    def score_single(
        self, 
        cnn_model: torch.nn.Module,
        behavioral_sequence: torch.Tensor
    ) -> dict[str, float]:
        """
        Score a single behavioral sequence.
        
        Computes reconstruction error and manifold-based scores for one
        behavioral pattern sequence.
        
        Parameters
        ----------
        cnn_model : torch.nn.Module
            Trained CNN autoencoder model
        behavioral_sequence : torch.Tensor
            Single behavioral sequence, shape (1, 1, time_steps, n_features)
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - 'reconstruction_error': CNN reconstruction MSE
            - 'off_manifold_distance': Normal deviation (β component)
            - 'density_score': k-NN density (if used)  
            - 'combined_score': Final anomaly score
        """
        cnn_model.eval()
        
        with torch.no_grad():
            # CNN forward pass
            reconstructed, latent = cnn_model(behavioral_sequence)
            
            # Reconstruction error
            recon_error = sequence_mse_2d(behavioral_sequence, reconstructed).item()
            
            # Manifold scores
            latent_np = latent.cpu().numpy().flatten()
            off_manifold_dist = self.manifold.normal_deviation(latent_np)
            if self.config.use_density:
                density_score = self.manifold.density_score(latent_np)
            else:
                density_score = 0.0
            
        # Combined scoring based on configuration
        if self.config.mode == "cnn_only":
            combined_score = self.config.alpha_cnn * recon_error
            
        elif self.config.mode == "manifold_only":
            combined_score = self.config.beta_manifold * off_manifold_dist
            if self.config.use_density:
                combined_score += self.config.gamma_density * density_score
                
        else:  # "cnn_plus_manifold" - the key test case
            combined_score = (
                self.config.alpha_cnn * recon_error + 
                self.config.beta_manifold * off_manifold_dist
            )
            if self.config.use_density:
                combined_score += self.config.gamma_density * density_score

        return {
            'reconstruction_error': recon_error,
            'off_manifold_distance': off_manifold_dist,  # The β component!
            'density_score': density_score,
            'combined_score': combined_score
        }

    def score_batch(
        self,
        cnn_model: torch.nn.Module,
        behavioral_sequences: torch.Tensor
    ) -> dict[str, np.ndarray]:
        """
        Score a batch of behavioral sequences efficiently.
        
        Parameters
        ----------
        cnn_model : torch.nn.Module
            Trained CNN autoencoder model  
        behavioral_sequences : torch.Tensor
            Batch of behavioral sequences, shape (batch_size, 1, time_steps, n_features)
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing arrays of scores:
            - 'reconstruction_error': CNN reconstruction MSEs, shape (batch_size,)
            - 'off_manifold_distance': Normal deviations, shape (batch_size,)
            - 'density_score': k-NN densities, shape (batch_size,) 
            - 'combined_score': Final anomaly scores, shape (batch_size,)
        """
        cnn_model.eval()
        batch_size = behavioral_sequences.shape[0]
        
        with torch.no_grad():
            # CNN forward pass for entire batch
            reconstructed, latents = cnn_model(behavioral_sequences)
            
            # Reconstruction errors for batch
            recon_errors = sequence_mse_2d(behavioral_sequences, reconstructed).cpu().numpy()
            
            # Manifold scores for each sequence
            off_manifold_dists = np.zeros(batch_size)
            density_scores = np.zeros(batch_size) if self.config.use_density else None
            
            latents_np = latents.cpu().numpy()
            for i in range(batch_size):
                off_manifold_dists[i] = self.manifold.normal_deviation(latents_np[i])
                if self.config.use_density:
                    density_scores[i] = self.manifold.density_score(latents_np[i])
        
        # Combined scoring for entire batch
        if self.config.mode == "cnn_only":
            combined_scores = self.config.alpha_cnn * recon_errors
            
        elif self.config.mode == "manifold_only":
            combined_scores = self.config.beta_manifold * off_manifold_dists
            if self.config.use_density and density_scores is not None:
                combined_scores += self.config.gamma_density * density_scores
                
        else:  # "cnn_plus_manifold"
            combined_scores = (
                self.config.alpha_cnn * recon_errors +
                self.config.beta_manifold * off_manifold_dists
            )
            if self.config.use_density and density_scores is not None:
                combined_scores += self.config.gamma_density * density_scores

        return {
            'reconstruction_error': recon_errors,
            'off_manifold_distance': off_manifold_dists,
            'density_score': density_scores if density_scores is not None else np.zeros(batch_size),
            'combined_score': combined_scores
        }

    def update_config(self, **kwargs) -> None:
        """
        Update scorer configuration.
        
        Useful for grid search over alpha, beta parameters.
        
        Parameters
        ----------
        **kwargs
            New configuration values (alpha_cnn, beta_manifold, etc.)
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")

    def get_config_summary(self) -> dict[str, any]:
        """
        Get current configuration summary.
        
        Returns
        -------
        Dict[str, any]
            Current scoring configuration
        """
        return {
            'mode': self.config.mode,
            'alpha_cnn': self.config.alpha_cnn,
            'beta_manifold': self.config.beta_manifold,  # The key parameter!
            'use_density': self.config.use_density,
            'gamma_density': self.config.gamma_density,
            'manifold_info': self.manifold.get_manifold_info()
        }


def compute_score_statistics(scores: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """
    Compute statistics for each score component.
    
    Useful for understanding the distribution and scale of different score components
    during hyperparameter tuning.
    
    Parameters
    ----------
    scores : Dict[str, np.ndarray]
        Score arrays from score_batch()
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Statistics for each score component
    """
    statistics = {}
    
    for score_name, score_array in scores.items():
        if isinstance(score_array, np.ndarray) and score_array.size > 0:
            statistics[score_name] = {
                'mean': float(np.mean(score_array)),
                'std': float(np.std(score_array)),
                'min': float(np.min(score_array)),
                'max': float(np.max(score_array)),
                'median': float(np.median(score_array)),
                'q25': float(np.percentile(score_array, 25)),
                'q75': float(np.percentile(score_array, 75))
            }
    
    return statistics


def normalize_scores(
    scores: dict[str, np.ndarray],
    reference_stats: dict[str, dict[str, float]] = None
) -> dict[str, np.ndarray]:
    """
    Normalize score components for fair combination.
    
    Parameters
    ----------
    scores : Dict[str, np.ndarray]
        Raw score arrays
    reference_stats : Dict[str, Dict[str, float]], optional
        Reference statistics for normalization (e.g., from training set)
        
    Returns
    -------
    Dict[str, np.ndarray]
        Normalized score arrays
    """
    normalized = {}
    
    for score_name, score_array in scores.items():
        if isinstance(score_array, np.ndarray) and score_array.size > 0:
            if reference_stats and score_name in reference_stats:
                # Normalize using reference statistics
                ref_mean = reference_stats[score_name]['mean']
                ref_std = reference_stats[score_name]['std']
                if ref_std > 1e-8:
                    normalized[score_name] = (score_array - ref_mean) / ref_std
                else:
                    normalized[score_name] = score_array - ref_mean
            else:
                # Z-score normalization
                score_mean = np.mean(score_array)
                score_std = np.std(score_array)
                if score_std > 1e-8:
                    normalized[score_name] = (score_array - score_mean) / score_std
                else:
                    normalized[score_name] = score_array - score_mean
        else:
            normalized[score_name] = score_array
    
    return normalized