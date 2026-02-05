"""
Dataset classes for UEBA behavioral pattern learning.

Provides different dataset configurations for various loss functions and
input representations.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def compute_stats(data_flat):
    """
    Compute mean and std for z-scoring normalization.
    
    Parameters
    ----------
    data_flat : np.ndarray
        Shape (N*T, F) - flattened data
        
    Returns
    -------
    mu : np.ndarray
        Mean per feature, shape (F,)
    sigma : np.ndarray
        Std per feature, shape (F,)
    """
    mu = data_flat.mean(axis=0)
    sigma = data_flat.std(axis=0) + 1e-10  # Avoid division by zero
    return mu, sigma


class SeqDataset(Dataset):
    """
    Standard dataset returning z-scored sequences.
    
    For baseline and standard MSE loss.
    
    Parameters
    ----------
    seq_array : np.ndarray
        Shape (N, T, F) - raw sequences
    mu : np.ndarray
        Shape (F,) - mean for z-scoring
    sigma : np.ndarray
        Shape (F,) - std for z-scoring
        
    Returns
    -------
    z_scored : torch.Tensor
        Shape (T, F) - z-scored sequence
    """
    
    def __init__(self, seq_array, mu, sigma):
        self.data = torch.tensor(seq_array, dtype=torch.float32)
        
        # Z-score normalize
        mu_tensor = torch.tensor(mu, dtype=torch.float32)
        sigma_tensor = torch.tensor(sigma, dtype=torch.float32)
        self.z_scored = (self.data - mu_tensor) / sigma_tensor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.z_scored[idx]


class WeightedSeqDataset(Dataset):
    """
    Dataset that returns both raw and z-scored sequences for weighted loss.
    
    For weighted MSE and two-term loss functions.
    
    Parameters
    ----------
    seq_array : np.ndarray
        Shape (N, T, F) - raw sequences
    mu : np.ndarray
        Shape (F,) - mean for z-scoring
    sigma : np.ndarray
        Shape (F,) - std for z-scoring
        
    Returns
    -------
    z_scored : torch.Tensor
        Shape (T, F) - z-scored sequence
    raw : torch.Tensor
        Shape (T, F) - raw sequence
    """
    
    def __init__(self, seq_array, mu, sigma):
        self.raw_data = torch.tensor(seq_array, dtype=torch.float32)
        
        # Z-score normalize
        mu_tensor = torch.tensor(mu, dtype=torch.float32)
        sigma_tensor = torch.tensor(sigma, dtype=torch.float32)
        self.z_scored = (self.raw_data - mu_tensor) / sigma_tensor
    
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        return self.z_scored[idx], self.raw_data[idx]


class MaskValueSeqDataset(Dataset):
    """
    Returns 2-channel input: [mask, z_scored_values], plus targets mask/value.
    
    For Ablation C: Mask + Value dual-channel architecture.
    
    Parameters
    ----------
    seq_array : np.ndarray
        Shape (N, T, F) - raw integer counts
    mu : np.ndarray
        Shape (F,) - mean for z-scoring, computed on training only
    sigma : np.ndarray
        Shape (F,) - std for z-scoring, computed on training only
        
    Returns
    -------
    x : torch.Tensor
        Shape (2, T, F) - stacked [mask, z_scored_values]
    mask : torch.Tensor
        Shape (1, T, F) - binary mask {0, 1}
    value : torch.Tensor
        Shape (1, T, F) - z-scored values
    """
    
    def __init__(self, seq_array: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> None:
        """
        Initialize dataset with raw sequences and normalization parameters.
        
        Parameters
        ----------
        seq_array : np.ndarray
            Raw integer counts, shape (N, T, F)
        mu : np.ndarray
            Mean for each feature, shape (F,)
        sigma : np.ndarray
            Standard deviation for each feature, shape (F,)
        """
        assert seq_array.ndim == 3, f"Expected 3D array, got {seq_array.ndim}D"
        self.raw = torch.tensor(seq_array, dtype=torch.float32)  # (N, T, F)

        mu_t = torch.tensor(mu, dtype=torch.float32).view(1, 1, -1)       # (1, 1, F)
        sigma_t = torch.tensor(sigma, dtype=torch.float32).view(1, 1, -1) # (1, 1, F)

        self.value = (self.raw - mu_t) / sigma_t  # (N, T, F) z-scored
        self.mask = (self.raw > 0).float()         # (N, T, F) binary {0, 1}

    def __len__(self) -> int:
        return self.raw.shape[0]

    def __getitem__(self, idx: int):
        """
        Get single sample.
        
        Returns
        -------
        x : torch.Tensor
            Shape (2, T, F) - [mask, value] stacked
        mask : torch.Tensor
            Shape (1, T, F) - binary mask
        value : torch.Tensor
            Shape (1, T, F) - z-scored values
        """
        mask = self.mask[idx].unsqueeze(0)    # (1, T, F)
        value = self.value[idx].unsqueeze(0)  # (1, T, F)
        x = torch.cat([mask, value], dim=0)   # (2, T, F)
        return x, mask, value


class TemporalPairedMaskValueSeqDataset(Dataset):
    """
    Returns pairs of consecutive windows for temporal regularization.
    
    For Block 2: Temporal consistency regularizer on normal consecutive windows.
    
    Builds pairs (t, t+1) from the same user, sorted chronologically.
    Used only for training on normal windows.
    
    Parameters
    ----------
    seq_array : np.ndarray
        Shape (N, T, F) - raw integer counts
    metadata : pd.DataFrame
        DataFrame with 'user_id' and 'window_start' columns
    mu : np.ndarray
        Shape (F,) - mean for z-scoring
    sigma : np.ndarray
        Shape (F,) - std for z-scoring
        
    Returns
    -------
    x_t : torch.Tensor
        Shape (2, T, F) - stacked [mask, z_scored_values] at time t
    mask_t : torch.Tensor
        Shape (1, T, F) - binary mask at time t
    value_t : torch.Tensor
        Shape (1, T, F) - z-scored values at time t
    x_t1 : torch.Tensor
        Shape (2, T, F) - stacked [mask, z_scored_values] at time t+1
    mask_t1 : torch.Tensor
        Shape (1, T, F) - binary mask at time t+1
    value_t1 : torch.Tensor
        Shape (1, T, F) - z-scored values at time t+1
    """
    
    def __init__(self, seq_array: np.ndarray, metadata, mu: np.ndarray, sigma: np.ndarray) -> None:
        """
        Initialize paired dataset by finding consecutive windows per user.
        
        Parameters
        ----------
        seq_array : np.ndarray
            Raw integer counts, shape (N, T, F)
        metadata : pd.DataFrame
            Must have 'user_id' and 'window_start' columns
        mu : np.ndarray
            Mean for each feature, shape (F,)
        sigma : np.ndarray
            Standard deviation for each feature, shape (F,)
        """
        import pandas as pd
        
        assert seq_array.ndim == 3, f"Expected 3D array, got {seq_array.ndim}D"
        assert len(seq_array) == len(metadata), f"Data ({len(seq_array)}) and metadata ({len(metadata)}) length mismatch"
        
        self.raw = torch.tensor(seq_array, dtype=torch.float32)  # (N, T, F)
        
        mu_t = torch.tensor(mu, dtype=torch.float32).view(1, 1, -1)       # (1, 1, F)
        sigma_t = torch.tensor(sigma, dtype=torch.float32).view(1, 1, -1) # (1, 1, F)
        
        self.value = (self.raw - mu_t) / sigma_t  # (N, T, F) z-scored
        self.mask = (self.raw > 0).float()         # (N, T, F) binary {0, 1}
        
        # Build consecutive pairs per user
        self.pairs = []  # List of (idx_t, idx_t1) tuples
        
        # Group by user and sort by time
        metadata = metadata.copy()
        metadata['idx'] = range(len(metadata))
        
        for user_id in metadata['user_id'].unique():
            user_mask = metadata['user_id'] == user_id
            user_meta = metadata[user_mask].copy()
            
            # Sort by window_start chronologically
            user_meta = user_meta.sort_values('window_start')
            user_indices = user_meta['idx'].values
            
            # Create consecutive pairs
            for i in range(len(user_indices) - 1):
                idx_t = user_indices[i]
                idx_t1 = user_indices[i + 1]
                self.pairs.append((idx_t, idx_t1))
        
        print(f"TemporalPairedMaskValueSeqDataset: created {len(self.pairs)} consecutive pairs from {len(seq_array)} windows")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int):
        """
        Get consecutive window pair.
        
        Returns
        -------
        x_t : torch.Tensor
            Shape (2, T, F) - [mask, value] at time t
        mask_t : torch.Tensor
            Shape (1, T, F) - binary mask at time t
        value_t : torch.Tensor
            Shape (1, T, F) - z-scored values at time t
        x_t1 : torch.Tensor
            Shape (2, T, F) - [mask, value] at time t+1
        mask_t1 : torch.Tensor
            Shape (1, T, F) - binary mask at time t+1
        value_t1 : torch.Tensor
            Shape (1, T, F) - z-scored values at time t+1
        """
        idx_t, idx_t1 = self.pairs[idx]
        
        # Get window at time t
        mask_t = self.mask[idx_t].unsqueeze(0)    # (1, T, F)
        value_t = self.value[idx_t].unsqueeze(0)  # (1, T, F)
        x_t = torch.cat([mask_t, value_t], dim=0)   # (2, T, F)
        
        # Get window at time t+1
        mask_t1 = self.mask[idx_t1].unsqueeze(0)    # (1, T, F)
        value_t1 = self.value[idx_t1].unsqueeze(0)  # (1, T, F)
        x_t1 = torch.cat([mask_t1, value_t1], dim=0)   # (2, T, F)
        
        return x_t, mask_t, value_t, x_t1, mask_t1, value_t1
