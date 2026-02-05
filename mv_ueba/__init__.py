"""Mask-value CNN UEBA package.

Dual-channel (mask + value) CNN autoencoder for behavioral anomaly detection,
with top-k aggregation for trajectory-level campaign detection.
"""

import logging

from mv_ueba.cnn_model import (
    UEBACNNAutoencoder,
    prepare_ueba_sequences_for_cnn,
    sequence_mse_2d,
    weighted_mse_loss,
    two_term_mse_loss,
    masked_value_mse,
    MaskValueLoss,
)
from mv_ueba.data import (
    SeqDataset,
    WeightedSeqDataset,
    MaskValueSeqDataset,
    TemporalPairedMaskValueSeqDataset,
    compute_stats,
)
from mv_ueba.latent_manifold import UEBALatentManifold, UEBAManifoldConfig
from mv_ueba.trajectory import TrajectoryAnalyzer, TrajectoryConfig

__all__ = [
    # CNN Autoencoder
    "UEBACNNAutoencoder",
    "sequence_mse_2d",
    "weighted_mse_loss",
    "two_term_mse_loss",
    "masked_value_mse",
    "MaskValueLoss",
    "prepare_ueba_sequences_for_cnn",
    # Latent manifold (k-NN, off-manifold distance)
    "UEBALatentManifold",
    "UEBAManifoldConfig",
    # Trajectory analysis (geodesic deviation)
    "TrajectoryAnalyzer",
    "TrajectoryConfig",
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
MAIN_LOGGER = logging.getLogger("mv_ueba")
MAIN_LOGGER.setLevel(logging.INFO)
if not MAIN_LOGGER.handlers:
    MAIN_LOGGER.addHandler(STREAM_HANDLER)
MAIN_LOGGER.propagate = False
