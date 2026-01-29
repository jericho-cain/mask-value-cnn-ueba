"""
Latent Manifold Geometry for UEBA Behavioral Pattern Detection

This module implements manifold learning in the CNN autoencoder latent space.
Builds k-NN graphs and computes local tangent space geometry for anomaly detection.

For UEBA:
- Training latents from normal behavioral patterns 
- Manifold captures "typical" user behavioral pattern structure
- Anomalous behaviors (insider threats, account takeover) appear off-manifold

Adapted from successful gravitational wave detection approach.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


@dataclass
class UEBAManifoldConfig:
    """
    Configuration for UEBA latent manifold construction.
    
    Parameters
    ----------
    k_neighbors : int, optional
        Number of neighbors for local geometry, by default 32
    tangent_dim : Optional[int], optional
        Intrinsic dimensionality of tangent space.
        If None, auto-estimate from PCA (95% variance), by default None
    metric : str, optional
        Distance metric for k-NN ('euclidean', 'cosine', etc.), by default 'euclidean'
    """
    k_neighbors: int = 32
    tangent_dim: int | None = 8  # Fixed conservative value (recommended for latent_dim=32)
    metric: str = "euclidean"


class UEBALatentManifold:
    """
    Latent Manifold models the geometry of the CNN autoencoder latent space.
    
    For UEBA, this manifold represents the structure of typical user behavioral patterns.
    Anomalous behaviors (coordinated attacks, insider threats, account takeover) should 
    appear as off-manifold deviations.
    
    The core insight: Normal behavioral patterns occupy a coherent geometric structure
    in the learned latent space. Anomalies violate this geometric structure.
    
    Key operations:
    - Builds k-NN index on training latents (normal behavioral patterns)
    - Estimates local tangent spaces via PCA
    - Computes normal deviation (off-manifold distance) - the β component
    - Computes density scores (k-NN distances)
    
    Parameters
    ----------
    train_latents : np.ndarray
        Training latent vectors from normal behavioral patterns, shape (N, d)
    config : UEBAManifoldConfig
        Configuration for manifold construction
        
    Attributes
    ----------
    train_latents : np.ndarray
        Training latent vectors from normal behavioral patterns
    config : UEBAManifoldConfig
        Configuration parameters
    _nn : NearestNeighbors
        k-NN index for efficient neighbor queries
        
    Examples
    --------
    >>> # Build manifold from normal behavioral pattern latents
    >>> config = UEBAManifoldConfig(k_neighbors=32)
    >>> manifold = UEBALatentManifold(train_latents, config)
    >>> 
    >>> # Score test behavioral pattern latent
    >>> off_manifold_score = manifold.normal_deviation(test_latent)
    >>> density_score = manifold.density_score(test_latent)
    """

    def __init__(self, train_latents: np.ndarray, config: UEBAManifoldConfig):
        """
        Initialize UEBA latent manifold from normal behavioral pattern training data.
        
        Parameters
        ----------
        train_latents : np.ndarray
            Training latent vectors from normal behavioral patterns, shape (N, d)
        config : UEBAManifoldConfig
            Configuration for manifold construction
        """
        assert train_latents.ndim == 2, "Expected (N, d) latent array"
        self.train_latents = train_latents.astype(np.float32)
        self.config = config

        self._nn = NearestNeighbors(
            n_neighbors=config.k_neighbors,
            metric=config.metric,
            algorithm='auto'
        )
        self._nn.fit(self.train_latents)

        # Auto-estimate tangent dimension if not provided
        if config.tangent_dim is None:
            self.tangent_dim = self._estimate_tangent_dim()
        else:
            self.tangent_dim = config.tangent_dim

    def _estimate_tangent_dim(self) -> int:
        """
        Auto-estimate intrinsic tangent space dimension.
        
        Uses PCA on training latents to find dimension needed for 95% variance.
        
        Returns
        -------
        int
            Estimated intrinsic dimension
        """
        pca = PCA()
        pca.fit(self.train_latents)
        
        # Find dimension for 95% of variance
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        tangent_dim = int(np.argmax(cumvar >= 0.95)) + 1
        
        # Reasonable bounds for UEBA
        tangent_dim = max(2, min(tangent_dim, self.train_latents.shape[1] // 2))
        
        return tangent_dim

    def _local_geometry(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute local manifold geometry at query point.
        
        Finds k-nearest neighbors and estimates local tangent space using PCA.
        
        Parameters
        ----------
        z : np.ndarray
            Query latent point, shape (d,)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            - mu: Local mean (centroid of neighborhood)
            - U: Tangent space basis vectors, shape (d, tangent_dim)  
            - distances: Distances to neighbors
        """
        z = np.asarray(z, dtype=np.float32).reshape(1, -1)
        
        # Find k nearest neighbors
        distances, indices = self._nn.kneighbors(z, return_distance=True)
        neighbors = self.train_latents[indices[0]]  # (k, d)
        
        # Local geometry
        mu = np.mean(neighbors, axis=0)  # (d,)
        centered = neighbors - mu        # (k, d)
        
        # Estimate tangent space with robust PCA
        if centered.shape[0] > 1 and not np.allclose(centered, 0):
            try:
                # Add small regularization for numerical stability
                if np.any(np.std(centered, axis=0) < 1e-8):
                    reg_noise = np.random.normal(0, 1e-6, centered.shape)
                    centered = centered + reg_noise
                
                pca = PCA(n_components=min(self.tangent_dim, centered.shape[0] - 1))
                pca.fit(centered)
                
                # Tangent space basis (d x tangent_dim)
                U = pca.components_.T  # (d, n_components)
                
                # Pad if needed
                if U.shape[1] < self.tangent_dim:
                    d = U.shape[0] 
                    padding = np.zeros((d, self.tangent_dim - U.shape[1]))
                    U = np.hstack([U, padding])
                    
            except (np.linalg.LinAlgError, ValueError):
                # Fallback: use identity for tangent space
                d = len(mu)
                U = np.eye(d)[:, :self.tangent_dim]
        else:
            # Degenerate neighborhood: use identity  
            d = len(mu)
            U = np.eye(d)[:, :self.tangent_dim]

        return mu, U, distances[0]

    def normal_deviation(self, z: np.ndarray) -> float:
        """
        Compute off-manifold distance (normal deviation) - the β component.
        
        This measures how far a behavioral pattern latent is from the normal
        behavioral pattern manifold. High values indicate behavioral anomalies
        that violate typical user behavioral geometry.
        
        **CRITICAL FIX:** Normalized by local neighborhood scale to make β
        consistent across dense/sparse regions of latent space.
        
        This is the key geometric score that enables manifold learning to detect
        anomalies that reconstruction error alone cannot capture.
        
        Parameters
        ----------
        z : np.ndarray
            Latent behavioral pattern, shape (d,)
            
        Returns
        -------
        float
            Scale-normalized off-manifold distance (normal deviation) - the β component
        """
        mu, U, distances = self._local_geometry(z)
        r = z - mu           # Vector from local mean to query point (d,)
        
        # Project onto tangent space and compute normal component
        r_tangent = U @ (U.T @ r)  # Tangent component
        r_normal = r - r_tangent   # Normal component (off-manifold)
        
        # Normalize by local neighborhood scale (median kNN distance)
        # This makes β consistent across dense/sparse regions
        local_scale = np.median(distances) if len(distances) > 0 else 1.0
        
        return float(np.linalg.norm(r_normal) / (local_scale + 1e-8))

    def density_score(self, z: np.ndarray) -> float:
        """
        Compute density score (mean k-NN distance).
        
        Simple k-NN-based density estimate: mean distance to neighbors.
        Lower distances = higher density (more typical behavioral patterns).
        Higher distances = lower density (more anomalous behavioral patterns).
        
        This provides an alternative geometric measure that can complement
        the normal deviation score.
        
        Parameters
        ----------
        z : np.ndarray
            Latent behavioral pattern, shape (d,)
            
        Returns
        -------
        float
            Mean distance to k nearest neighbors
        """
        z = np.asarray(z, dtype=np.float32).reshape(1, -1)
        distances, _ = self._nn.kneighbors(z, return_distance=True)
        
        # Return mean distance (excluding potential self-match)
        if distances[0, 0] < 1e-10:  # Very close to training point
            return float(np.mean(distances[0, 1:]))
        else:
            return float(np.mean(distances[0]))

    def get_manifold_info(self) -> dict:
        """
        Get information about the constructed manifold.
        
        Returns
        -------
        dict
            Dictionary with manifold statistics
        """
        return {
            'n_training_points': len(self.train_latents),
            'latent_dim': self.train_latents.shape[1],
            'k_neighbors': self.config.k_neighbors,
            'tangent_dim': self.tangent_dim,
            'metric': self.config.metric,
            'manifold_type': 'UEBA Behavioral Pattern Manifold'
        }

    def validate_manifold_quality(self) -> dict:
        """
        Validate the quality of the constructed manifold.
        
        Computes metrics to assess whether the training data forms
        a coherent manifold structure suitable for anomaly detection.
        
        Returns
        -------
        dict
            Validation metrics including:
            - avg_neighbor_distance: Average distance between neighbors
            - manifold_coherence: How well PCA captures local structure
            - density_variance: Variation in local densities
        """
        n_samples = min(100, len(self.train_latents))  # Sample for efficiency
        indices = np.random.choice(len(self.train_latents), n_samples, replace=False)
        
        neighbor_distances = []
        pca_explained_vars = []
        density_scores = []
        
        for i in indices:
            point = self.train_latents[i]
            
            # Get local geometry
            mu, U, distances = self._local_geometry(point)
            neighbor_distances.append(np.mean(distances))
            
            # Compute how much variance the tangent space captures
            # Get k+1 neighbors and exclude self if present
            distances_val, indices_val = self._nn.kneighbors([point], n_neighbors=self.config.k_neighbors+1, return_distance=True)
            # Exclude self-match (distance < 1e-10)
            if distances_val[0, 0] < 1e-10:
                neighbors = self.train_latents[indices_val[0, 1:]]  # Exclude first (self)
            else:
                neighbors = self.train_latents[indices_val[0, :-1]]  # Exclude last
            centered = neighbors - mu
            if not np.allclose(centered, 0):
                try:
                    pca = PCA()
                    pca.fit(centered)
                    explained_var = np.sum(pca.explained_variance_ratio_[:self.tangent_dim])
                    pca_explained_vars.append(explained_var)
                except (np.linalg.LinAlgError, ValueError):
                    pca_explained_vars.append(0.5)  # Default
            
            # Compute density
            density_scores.append(self.density_score(point))
        
        return {
            'avg_neighbor_distance': float(np.mean(neighbor_distances)),
            'neighbor_distance_std': float(np.std(neighbor_distances)),
            'manifold_coherence': float(np.mean(pca_explained_vars)) if pca_explained_vars else 0.5,
            'density_variance': float(np.var(density_scores)),
            'n_validation_points': n_samples
        }
    
    def save(self, filepath: str) -> None:
        """
        Save manifold structure to file.
        
        Parameters
        ----------
        filepath : str
            Path to save manifold data
        """
        np.savez_compressed(
            filepath,
            train_latents=self.train_latents,
            k_neighbors=self.config.k_neighbors,
            tangent_dim=self.tangent_dim,
            metric=self.config.metric
        )
    
    @classmethod
    def load(cls, filepath: str) -> 'UEBALatentManifold':
        """
        Load manifold structure from file.
        
        Parameters
        ----------
        filepath : str
            Path to load manifold data from
            
        Returns
        -------
        UEBALatentManifold
            Loaded manifold instance
        """
        data = np.load(filepath)
        
        config = UEBAManifoldConfig(
            k_neighbors=int(data['k_neighbors']),
            tangent_dim=int(data['tangent_dim']) if 'tangent_dim' in data else None,
            metric=str(data['metric']) if 'metric' in data else 'euclidean'
        )
        
        manifold = cls(data['train_latents'], config)
        return manifold