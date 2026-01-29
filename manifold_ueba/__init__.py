"""Manifold UEBA model package.

Geometric anomaly detection for user and entity behavior analytics using
manifold learning on CNN autoencoder latent spaces.
"""

import logging

from manifold_ueba.cnn_model import (
    UEBACNNAutoencoder,
    prepare_ueba_sequences_for_cnn,
    sequence_mse_2d,
    weighted_mse_loss,
    two_term_mse_loss,
    masked_value_mse,
    MaskValueLoss,
)
from manifold_ueba.data import (
    SeqDataset,
    WeightedSeqDataset,
    MaskValueSeqDataset,
    TemporalPairedMaskValueSeqDataset,
    compute_stats,
)
from manifold_ueba.grid_search import GridSearchResult, grid_search_alpha_beta
from manifold_ueba.latent_manifold import UEBALatentManifold, UEBAManifoldConfig
from manifold_ueba.manifold_scorer import UEBAManifoldScorer, UEBAManifoldScorerConfig
from manifold_ueba.trajectory import TrajectoryAnalyzer, TrajectoryConfig, create_trajectories_from_sequences

__all__ = [
    # CNN Autoencoder
    "UEBACNNAutoencoder",
    "sequence_mse_2d",
    "weighted_mse_loss",
    "two_term_mse_loss",
    "masked_value_mse",
    "MaskValueLoss",
    "prepare_ueba_sequences_for_cnn",
    # Manifold Learning
    "UEBALatentManifold",
    "UEBAManifoldConfig",
    # Scoring
    "UEBAManifoldScorer",
    "UEBAManifoldScorerConfig",
    "grid_search_alpha_beta",
    "GridSearchResult",
    # Trajectory Analysis
    "TrajectoryAnalyzer",
    "TrajectoryConfig",
    "create_trajectories_from_sequences",
    # Data utilities
    "SeqDataset",
    "WeightedSeqDataset",
    "MaskValueSeqDataset",
    "TemporalPairedMaskValueSeqDataset",
    "compute_stats",
]

__version__ = "0.1.0"


# Logging setup
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.INFO)
FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
STREAM_HANDLER.setFormatter(FORMATTER)
MAIN_LOGGER = logging.getLogger("manifold_ueba")
MAIN_LOGGER.setLevel(logging.INFO)
if not MAIN_LOGGER.handlers:
    MAIN_LOGGER.addHandler(STREAM_HANDLER)
MAIN_LOGGER.propagate = False
