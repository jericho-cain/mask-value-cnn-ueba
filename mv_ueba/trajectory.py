"""
Trajectory Analysis for UEBA Manifold Learning

This module implements geodesic deviation analysis for detecting anomalous
behavioral trajectories through the latent manifold.

Mathematical Framework:
-----------------------
Given a learned manifold M from normal behavior, we analyze trajectories
γ(t) = {z₁, z₂, ..., zₙ} through the latent space.

For normal users, trajectories should approximately follow geodesics on M.
Anomalous behavior manifests as geodesic deviation - trajectories that
diverge from expected paths.

Key metrics:
1. Velocity: v_i = z_{i+1} - z_i (tangent vector)
2. Acceleration: a_i = v_{i+1} - v_i (geodesic deviation in flat space)
3. Off-manifold drift: cumulative perpendicular distance over trajectory
4. Path irregularity: deviation from locally-predicted next position

References:
- Geodesic deviation equation (Jacobi equation) from differential geometry
- Parallel transport approximation via Schild's ladder
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory analysis."""
    
    # Manifold reference
    k_neighbors: int = 16  # k for local geodesic estimation
    
    # Trajectory parameters
    min_trajectory_length: int = 4  # Minimum points for analysis
    
    # Anomaly thresholds (learned from data or set manually)
    velocity_threshold: Optional[float] = None
    acceleration_threshold: Optional[float] = None
    deviation_threshold: Optional[float] = None


class TrajectoryAnalyzer:
    """
    Analyzes trajectories through latent space for geodesic deviation.
    
    Given a manifold built from normal behavior, this class:
    1. Tracks sequences of latent points (trajectories)
    2. Computes kinematic features (velocity, acceleration)
    3. Measures deviation from expected geodesic paths
    4. Produces anomaly scores for trajectories
    """
    
    def __init__(
        self,
        manifold,  # UEBALatentManifold instance
        config: Optional[TrajectoryConfig] = None
    ):
        """
        Initialize trajectory analyzer with a reference manifold.
        
        Args:
            manifold: Trained UEBALatentManifold containing normal behavior
            config: Configuration parameters
        """
        self.manifold = manifold
        self.config = config or TrajectoryConfig()
        self.reference_stats = None
        
    def compute_trajectory_features(
        self,
        trajectory: np.ndarray
    ) -> dict:
        """
        Compute kinematic and geometric features for a trajectory.
        
        Args:
            trajectory: Array of shape (n_points, latent_dim) representing
                       a path through latent space
                       
        Returns:
            Dictionary of trajectory features
        """
        n_points = len(trajectory)
        
        if n_points < self.config.min_trajectory_length:
            return {
                'valid': False,
                'reason': f'Trajectory too short: {n_points} < {self.config.min_trajectory_length}'
            }
        
        # Velocity vectors (tangent to trajectory)
        velocities = np.diff(trajectory, axis=0)  # (n-1, latent_dim)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        # Acceleration vectors (geodesic deviation in flat space)
        accelerations = np.diff(velocities, axis=0)  # (n-2, latent_dim)
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        # Off-manifold distance at each point
        manifold_distances = np.array([
            self.manifold.normal_deviation(z) for z in trajectory
        ])
        
        # Cumulative path length
        path_length = np.sum(velocity_magnitudes)
        
        # Displacement (start to end)
        displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
        
        # Tortuosity: path_length / displacement (1.0 = straight line)
        tortuosity = path_length / (displacement + 1e-8)
        
        # Direction changes (cosine of angle between consecutive velocities)
        direction_changes = []
        for i in range(len(velocities) - 1):
            v1, v2 = velocities[i], velocities[i + 1]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            direction_changes.append(cos_angle)
        direction_changes = np.array(direction_changes)
        
        # Off-manifold drift: is the trajectory moving away from the manifold?
        manifold_drift = manifold_distances[-1] - manifold_distances[0]
        manifold_drift_rate = manifold_drift / n_points
        
        return {
            'valid': True,
            'n_points': n_points,
            
            # Velocity statistics
            'mean_velocity': np.mean(velocity_magnitudes),
            'max_velocity': np.max(velocity_magnitudes),
            'std_velocity': np.std(velocity_magnitudes),
            
            # Acceleration statistics (geodesic deviation proxy)
            'mean_acceleration': np.mean(acceleration_magnitudes),
            'max_acceleration': np.max(acceleration_magnitudes),
            'std_acceleration': np.std(acceleration_magnitudes),
            
            # Manifold distance statistics
            'mean_manifold_distance': np.mean(manifold_distances),
            'max_manifold_distance': np.max(manifold_distances),
            'manifold_drift': manifold_drift,
            'manifold_drift_rate': manifold_drift_rate,
            
            # Path geometry
            'path_length': path_length,
            'displacement': displacement,
            'tortuosity': tortuosity,
            
            # Direction statistics
            'mean_direction_change': np.mean(direction_changes),
            'min_direction_change': np.min(direction_changes),  # Most abrupt turn
            
            # Raw arrays for detailed analysis
            'velocities': velocities,
            'accelerations': accelerations,
            'manifold_distances': manifold_distances,
            'direction_changes': direction_changes,
        }
    
    def fit_reference_statistics(
        self,
        normal_trajectories: list[np.ndarray]
    ) -> None:
        """
        Compute reference statistics from normal trajectories.
        
        These statistics define what "normal" trajectories look like,
        enabling anomaly detection on new trajectories.
        
        Args:
            normal_trajectories: List of trajectory arrays from normal users
        """
        all_features = []
        
        for traj in normal_trajectories:
            features = self.compute_trajectory_features(traj)
            if features['valid']:
                all_features.append(features)
        
        if len(all_features) == 0:
            raise ValueError("No valid trajectories for reference statistics")
        
        # Compute statistics for each feature
        feature_names = [
            'mean_velocity', 'max_velocity', 'mean_acceleration', 'max_acceleration',
            'mean_manifold_distance', 'max_manifold_distance', 'manifold_drift_rate',
            'tortuosity', 'mean_direction_change', 'min_direction_change'
        ]
        
        self.reference_stats = {}
        for name in feature_names:
            values = [f[name] for f in all_features]
            self.reference_stats[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99),
            }
        
        logger.info(f"Computed reference statistics from {len(all_features)} trajectories")
    
    def score_trajectory(
        self,
        trajectory: np.ndarray,
        return_features: bool = False
    ) -> float | tuple[float, dict]:
        """
        Compute anomaly score for a trajectory.
        
        Score is based on geodesic deviation: how much the trajectory
        deviates from expected normal behavior paths.
        
        Args:
            trajectory: Array of shape (n_points, latent_dim)
            return_features: If True, also return feature dictionary
            
        Returns:
            Anomaly score (higher = more anomalous)
            Optionally: (score, features) tuple
        """
        features = self.compute_trajectory_features(trajectory)
        
        if not features['valid']:
            score = 0.0 if return_features else 0.0
            return (score, features) if return_features else score
        
        if self.reference_stats is None:
            # No reference - return raw acceleration as proxy
            score = features['mean_acceleration'] + features['max_manifold_distance']
            return (score, features) if return_features else score
        
        # Compute z-scores for key features relative to normal
        anomaly_components = []
        
        # Acceleration (geodesic deviation)
        acc_z = (features['mean_acceleration'] - self.reference_stats['mean_acceleration']['mean']) / \
                (self.reference_stats['mean_acceleration']['std'] + 1e-8)
        anomaly_components.append(max(0, acc_z))  # Only penalize high acceleration
        
        # Manifold drift (moving away from normal manifold)
        drift_z = (features['manifold_drift_rate'] - self.reference_stats['manifold_drift_rate']['mean']) / \
                  (self.reference_stats['manifold_drift_rate']['std'] + 1e-8)
        anomaly_components.append(max(0, drift_z))  # Only penalize positive drift
        
        # Maximum manifold distance
        dist_z = (features['max_manifold_distance'] - self.reference_stats['max_manifold_distance']['mean']) / \
                 (self.reference_stats['max_manifold_distance']['std'] + 1e-8)
        anomaly_components.append(max(0, dist_z))
        
        # Tortuosity (erratic path)
        tort_z = (features['tortuosity'] - self.reference_stats['tortuosity']['mean']) / \
                 (self.reference_stats['tortuosity']['std'] + 1e-8)
        anomaly_components.append(max(0, tort_z))
        
        # Abrupt direction changes
        dir_z = (self.reference_stats['min_direction_change']['mean'] - features['min_direction_change']) / \
                (self.reference_stats['min_direction_change']['std'] + 1e-8)
        anomaly_components.append(max(0, dir_z))  # Lower cosine = more abrupt
        
        # Combined score (could weight these differently)
        score = np.mean(anomaly_components)
        
        return (score, features) if return_features else score
    
    def analyze_trajectory_sequence(
        self,
        latent_points: np.ndarray,
        window_size: int = 6,
        stride: int = 1
    ) -> list[dict]:
        """
        Analyze a long sequence using sliding window trajectories.
        
        Args:
            latent_points: Array of shape (n_total_points, latent_dim)
            window_size: Number of points per trajectory window
            stride: Step size between windows
            
        Returns:
            List of analysis results for each window
        """
        results = []
        n_points = len(latent_points)
        
        for start in range(0, n_points - window_size + 1, stride):
            end = start + window_size
            window_traj = latent_points[start:end]
            
            score, features = self.score_trajectory(window_traj, return_features=True)
            
            results.append({
                'start_idx': start,
                'end_idx': end,
                'score': score,
                'features': features
            })
        
        return results


def create_trajectories_from_sequences(
    latent_sequences: np.ndarray,
    user_ids: np.ndarray,
    sequence_length: int = 6
) -> dict[str, list[np.ndarray]]:
    """
    Group consecutive latent points into trajectories by user.
    
    Args:
        latent_sequences: Array of latent vectors (n_sequences, latent_dim)
        user_ids: Array of user IDs for each sequence
        sequence_length: Number of consecutive points per trajectory
        
    Returns:
        Dictionary mapping user_id to list of trajectory arrays
    """
    user_trajectories = {}
    
    unique_users = np.unique(user_ids)
    
    for user in unique_users:
        user_mask = user_ids == user
        user_latents = latent_sequences[user_mask]
        
        trajectories = []
        for i in range(0, len(user_latents) - sequence_length + 1, sequence_length):
            traj = user_latents[i:i + sequence_length]
            trajectories.append(traj)
        
        if trajectories:
            user_trajectories[user] = trajectories
    
    return user_trajectories
