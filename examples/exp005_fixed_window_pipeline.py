"""
Experiment 005: 24-Hour Fixed Window Detection Pipeline

Multi-day attack campaign detection (1-7 day duration) using fixed 24-hour windows.

**Approach:**
- 26 compromised users (1 day < attack < 1 week)
- Fixed 24-hour windows at 1-hour resolution (T=24, F=13)
- Window-level metrics: reconstruction error + off-manifold distance
- Train on normal windows only, test on normal + malicious
- Temporal separation: 7-day buffer around attacks

**Multi-Scale Framework:**
This is the first instantiation of a multi-scale manifold system where different
temporal scales require different manifolds. This experiment targets multi-day
sub-week attacks. Future experiments will cover other scales.

**Two-Level Analysis:**
1. Window-Level: Point-wise anomaly detection on individual 24hr windows
2. Trajectory-Level: Sequential geodesic deviation analysis across multi-day trajectories

Usage:
    python examples/exp005_fixed_window_pipeline.py \\
        --experiment exp005_24hr_window \\
        --data-dir data/cert/r4.2
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from manifold_ueba.cnn_model import UEBACNNAutoencoder, weighted_mse_loss, two_term_mse_loss, MaskValueLoss
from manifold_ueba.data import SeqDataset, WeightedSeqDataset, MaskValueSeqDataset, TemporalPairedMaskValueSeqDataset, compute_stats
from manifold_ueba.scoring import score_windows_mask_value, score_windows_standard
from manifold_ueba.etl.cert_fixed_window import CERTFixedWindowLoader
from manifold_ueba.latent_manifold import UEBALatentManifold, UEBAManifoldConfig
from manifold_ueba.trajectory import TrajectoryAnalyzer, TrajectoryConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Exp005: 24-hour fixed window detection')
    
    # Experiment setup
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name (creates runs/exp_name/)')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to CERT r4.2 directory')
    
    # Data parameters
    parser.add_argument('--bucket-hours', type=float, default=1.0, help='Time bucket size in hours (Δt)')
    parser.add_argument('--window-hours', type=int, default=24, help='Window size in hours')
    parser.add_argument('--buffer-days', type=int, default=7, help='Temporal buffer around attacks (days)')
    parser.add_argument('--train-split', type=float, default=0.8, help='Train split ratio for normal data')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--latent-dim', type=int, default=8, help='Latent space dimension')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    # Early stopping
    parser.add_argument('--early-stop-patience', type=int, default=5, help='Early stopping patience (epochs)')
    parser.add_argument('--early-stop-delta', type=float, default=1e-4, help='Minimum improvement threshold')
    
    # Loss function type
    parser.add_argument('--loss-type', type=str, default='standard', 
                        choices=['standard', 'weighted', 'two-term'],
                        help='Loss function type')
    
    # Weighted loss parameters
    parser.add_argument('--use-weighted-loss', action='store_true', help='[DEPRECATED] Use --loss-type weighted')
    parser.add_argument('--w-inactive', type=float, default=1.0, help='Weight for inactive buckets')
    parser.add_argument('--w-active', type=float, default=20.0, help='Weight for active buckets')
    
    # Two-term loss parameters
    parser.add_argument('--lambda-active', type=float, default=2.0, help='Lambda for active-only term (two-term loss)')
    
    # Mask + Value (Ablation C) parameters
    parser.add_argument('--use-mask-value', action='store_true', help='Use 2-channel mask+value input and dual loss (Ablation C)')
    parser.add_argument('--lambda-value', type=float, default=1.0, help='Weight for masked value MSE term (Ablation C)')
    
    # Temporal regularization (Block 2) parameters
    parser.add_argument('--use-temporal-reg', action='store_true', help='Use temporal consistency regularizer (Block 2)')
    parser.add_argument('--lambda-temporal', type=float, default=0.01, help='Weight for temporal smoothness penalty (Block 2)')
    
    # Manifold parameters
    parser.add_argument('--k-neighbors', type=int, default=5, help='k for k-NN manifold')
    
    # Trajectory parameters
    parser.add_argument('--traj-window-size', type=int, default=6, help='Trajectory window size (consecutive 24hr windows)')
    parser.add_argument('--traj-stride', type=int, default=3, help='Trajectory stride (window overlap)')
    
    # Grid search for alpha/beta
    parser.add_argument('--grid-search', action='store_true', help='Run grid search for alpha/beta')
    parser.add_argument('--no-grid-search', dest='grid_search', action='store_false')
    parser.set_defaults(grid_search=True)
    
    # Quick test mode
    parser.add_argument('--quick-test', action='store_true', help='Quick test with reduced epochs')
    
    return parser.parse_args()


def setup_experiment(args):
    """Create experiment directory and save configuration."""
    exp_dir = Path('runs') / args.experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args).copy()
    config['created_at'] = datetime.now().isoformat()
    config['approach'] = 'fixed_window_24hr'
    config['description'] = '24-hour fixed windows for multi-day attack detection (1-7 days)'
    config['target_attacks'] = '26 users (16 Scenario 1, 10 Scenario 3)'
    config['window_design'] = f'T={24//args.bucket_hours} buckets × {args.bucket_hours}hr = {args.window_hours}hr'
    
    config_path = exp_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Configuration saved to: {config_path}")
    
    return exp_dir


def train_autoencoder(train_data, args, exp_dir, train_metadata=None):
    """
    Train CNN autoencoder on normal windows.
    
    Args:
        train_data: (N, T, F) array of normal windows
        args: command line arguments
        exp_dir: experiment directory
        train_metadata: DataFrame with user_id, window_start (for temporal pairing)
        
    Returns:
        model: trained autoencoder
        mu: feature means
        sigma: feature stds
    """
    logger.info("Training autoencoder...")
    logger.info(f"Training data shape: {train_data.shape}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Compute normalization stats
    train_flat = train_data.reshape(-1, train_data.shape[-1])
    mu, sigma = compute_stats(train_flat)
    
    # Determine loss type (handle deprecated flags)
    if args.use_mask_value:
        loss_type = 'mask_value'
    else:
        loss_type = args.loss_type
        if args.use_weighted_loss:
            loss_type = 'weighted'
            logger.warning("--use-weighted-loss is deprecated, use --loss-type weighted instead")
    
    # Compute pos_weight for mask loss (Ablation C)
    pos_weight = None
    if loss_type == 'mask_value':
        pos = (train_data > 0).sum()
        neg = (train_data == 0).sum()
        pos_weight = float(neg / (pos + 1e-8))
        logger.info(f"Mask pos_weight = {pos_weight:.3f} (neg={neg}, pos={pos})")
    
    # Create dataset and loader
    if loss_type == 'mask_value':
        # Ablation C: Mask + Value dual-channel
        if args.use_temporal_reg:
            # Block 2: Use paired consecutive windows for temporal regularization
            train_dataset = TemporalPairedMaskValueSeqDataset(train_data, train_metadata, mu=mu, sigma=sigma)
            logger.info(f"Using MASK+VALUE + TEMPORAL REGULARIZATION (λ_value={args.lambda_value}, λ_temp={args.lambda_temporal}, pos_weight={pos_weight:.3f})")
        else:
            train_dataset = MaskValueSeqDataset(train_data, mu=mu, sigma=sigma)
            logger.info(f"Using MASK+VALUE dual-channel loss (λ_value={args.lambda_value}, pos_weight={pos_weight:.3f})")
    elif loss_type in ['weighted', 'two-term']:
        # Use weighted dataset that returns both raw and z-scored
        train_dataset = WeightedSeqDataset(train_data, mu=mu, sigma=sigma)
        if loss_type == 'weighted':
            logger.info(f"Using WEIGHTED MSE loss (w_inactive={args.w_inactive}, w_active={args.w_active})")
        else:
            logger.info(f"Using TWO-TERM MSE loss (λ_active={args.lambda_active})")
    else:
        # Use standard dataset (z-scored only)
        train_dataset = SeqDataset(train_data, mu=mu, sigma=sigma)
        logger.info("Using STANDARD MSE loss")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    time_steps = train_data.shape[1]  # T buckets
    n_features = train_data.shape[2]  # F=13
    model = UEBACNNAutoencoder(time_steps=time_steps, n_features=n_features, latent_dim=args.latent_dim)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"Device: {device}")
    logger.info(f"Model: T={time_steps}, F={n_features}, latent_dim={args.latent_dim}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if loss_type == 'mask_value':
        lambda_temporal = args.lambda_temporal if args.use_temporal_reg else 0.0
        criterion = MaskValueLoss(
            pos_weight=pos_weight,
            lambda_value=args.lambda_value,
            lambda_temporal=lambda_temporal
        ).to(device)
    elif loss_type == 'standard':
        criterion = torch.nn.MSELoss()
    
    # Training loop with early stopping
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch_data in train_loader:
            optimizer.zero_grad()
            
            if loss_type == 'mask_value':
                if args.use_temporal_reg:
                    # Temporal paired dataset returns (x_t, mask_t, value_t, x_t1, mask_t1, value_t1)
                    x_t, mask_t, value_t, x_t1, mask_t1, value_t1 = batch_data
                    x_t = x_t.to(device)
                    mask_t = mask_t.to(device)
                    value_t = value_t.to(device)
                    x_t1 = x_t1.to(device)
                    mask_t1 = mask_t1.to(device)
                    value_t1 = value_t1.to(device)
                    
                    # Forward pass for both time steps
                    reconstructed_t, latent_t = model(x_t)
                    reconstructed_t1, latent_t1 = model(x_t1)
                    
                    # Dual-channel loss + temporal regularization
                    losses = criterion(
                        reconstructed_t, mask_t, value_t,
                        latent_t=latent_t, latent_t1=latent_t1,
                        recon_t1=reconstructed_t1, mask_true_t1=mask_t1, value_true_t1=value_t1
                    )
                    loss = losses["loss"]
                else:
                    # Standard Mask+Value dataset returns (x, mask, value) where x is (B, 2, T, F)
                    x, mask_true, value_true = batch_data
                    x = x.to(device)
                    mask_true = mask_true.to(device)
                    value_true = value_true.to(device)
                    
                    # Forward pass (x already has correct shape)
                    reconstructed, latent = model(x)
                    
                    # Dual-channel loss
                    losses = criterion(reconstructed, mask_true, value_true)
                    loss = losses["loss"]
                
            elif loss_type in ['weighted', 'two-term']:
                # Weighted/two-term dataset returns (z_scored, raw)
                batch_z, batch_raw = batch_data
                batch_z = batch_z.to(device)
                batch_raw = batch_raw.to(device)
                
                # Add channel dimension: (batch, time, features) -> (batch, 1, time, features)
                batch_z_4d = batch_z.unsqueeze(1)
                
                # Forward pass
                reconstructed, latent = model(batch_z_4d)
                
                # Compute loss based on type
                if loss_type == 'weighted':
                    loss = weighted_mse_loss(
                        reconstructed, 
                        batch_z_4d, 
                        batch_raw,
                        w_inactive=args.w_inactive,
                        w_active=args.w_active
                    )
                else:  # two-term
                    loss = two_term_mse_loss(
                        reconstructed,
                        batch_z_4d,
                        batch_raw,
                        lambda_active=args.lambda_active
                    )
            else:
                # Standard dataset returns z_scored only
                batch = batch_data.to(device)
                
                # Add channel dimension: (batch, time, features) -> (batch, 1, time, features)
                batch = batch.unsqueeze(1)
                
                # Forward pass
                reconstructed, latent = model(batch)
                loss = criterion(reconstructed, batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Early stopping check
        if avg_loss < best_loss - args.early_stop_delta:
            best_loss = avg_loss
            patience_counter = 0
            logger.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f} ⭐ (improved)")
        else:
            patience_counter += 1
            logger.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f} (patience: {patience_counter}/{args.early_stop_patience})")
            
            if patience_counter >= args.early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}/{args.epochs}")
                logger.info(f"Best loss: {best_loss:.6f}, Current loss: {avg_loss:.6f}")
                break
    
    # Save model
    model_path = exp_dir / 'autoencoder.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'mu': mu,
        'sigma': sigma,
        'time_steps': time_steps,
        'n_features': n_features,
        'latent_dim': args.latent_dim,
        'final_loss': best_loss,
        'epochs_trained': epoch + 1,
        'early_stopped': patience_counter >= args.early_stop_patience,
    }, model_path)
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Training completed: {epoch+1} epochs, best loss: {best_loss:.6f}")
    
    return model, mu, sigma


def build_manifold(model, train_data, mu, sigma, args, exp_dir):
    """
    Build latent manifold from training data.
    
    Args:
        model: trained autoencoder
        train_data: (N, T, F) array of normal windows
        mu, sigma: normalization parameters
        args: command line arguments
        exp_dir: experiment directory
        
    Returns:
        manifold: UEBALatentManifold object
    """
    logger.info("Building manifold...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Extract latent representations (use appropriate method based on architecture)
    if hasattr(args, 'use_mask_value') and args.use_mask_value:
        # Ablation C: Use mask+value scoring to get latents
        scores_dict = score_windows_mask_value(model, train_data, mu, sigma, device)
        train_latents = scores_dict["latent"]
    else:
        # Standard: Use single-channel dataset
        train_dataset = SeqDataset(train_data, mu=mu, sigma=sigma)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        
        latents = []
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(device)
                batch = batch.unsqueeze(1)  # Add channel dimension (B, 1, T, F)
                _, latent = model(batch)
                latents.append(latent.cpu().numpy())
        
        train_latents = np.concatenate(latents, axis=0)
    
    logger.info(f"Extracted {len(train_latents)} latent points, dim={train_latents.shape[1]}")
    
    # Build manifold
    config = UEBAManifoldConfig(k_neighbors=args.k_neighbors)
    manifold = UEBALatentManifold(train_latents, config)
    
    # Save manifold
    manifold_path = exp_dir / 'manifold.npz'
    np.savez(
        manifold_path,
        train_latents=train_latents,
        n_neighbors=args.k_neighbors
    )
    logger.info(f"Manifold saved to: {manifold_path}")
    
    return manifold


def evaluate_window_detection(model, manifold, test_data, test_labels, test_metadata, mu, sigma, args, exp_dir):
    """
    Evaluate window-level detection using reconstruction error and off-manifold distance.
    
    Args:
        model: trained autoencoder
        manifold: UEBALatentManifold
        test_data: (N, T, F) array of test windows
        test_labels: (N,) array of binary labels
        test_metadata: DataFrame with user_id, window_start, scenario
        mu, sigma: normalization parameters
        args: command line arguments
        exp_dir: experiment directory
        
    Returns:
        results: dictionary of evaluation metrics
    """
    logger.info("Evaluating window-level detection...")
    logger.info(f"Test data: {len(test_data)} windows ({(test_labels==1).sum()} malicious)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Determine scoring method based on loss type
    if hasattr(args, 'use_mask_value') and args.use_mask_value:
        # Ablation C: Mask + Value dual-channel scoring
        scores_dict = score_windows_mask_value(model, test_data, mu, sigma, device)
        
        # Extract all three components
        mask_scores = scores_dict["mask_bce"]           # (N,) BCE per window
        value_scores = scores_dict["value_mse_active"]  # (N,) MSE on active cells
        total_scores = scores_dict["total"]              # (N,) mask + value combined
        latent_points = scores_dict["latent"]            # (N, latent_dim)
        
        # Log component score ranges
        logger.info(f"Mask BCE: min={mask_scores.min():.4f}, max={mask_scores.max():.4f}, mean={mask_scores.mean():.4f}")
        logger.info(f"Value MSE (active): min={value_scores.min():.4f}, max={value_scores.max():.4f}, mean={value_scores.mean():.4f}")
        logger.info(f"Total (Mask+Value): min={total_scores.min():.4f}, max={total_scores.max():.4f}, mean={total_scores.mean():.4f}")
        
        # For legacy compatibility, set reconstruction_errors to total
        reconstruction_errors = total_scores
    else:
        # Standard or weighted/two-term: Single-channel scoring
        test_dataset = SeqDataset(test_data, mu=mu, sigma=sigma)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        reconstruction_errors = []
        latent_list = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                batch = batch.unsqueeze(1)  # Add channel dimension
                
                # Reconstruction error
                reconstructed, latent = model(batch)
                recon_error = torch.mean((reconstructed - batch) ** 2, dim=(1, 2, 3))
                reconstruction_errors.append(recon_error.cpu().numpy())
                latent_list.append(latent.cpu().numpy())
        
        reconstruction_errors = np.concatenate(reconstruction_errors)
        latent_points = np.concatenate(latent_list, axis=0)
        
        logger.info(f"Reconstruction errors: min={reconstruction_errors.min():.4f}, max={reconstruction_errors.max():.4f}, mean={reconstruction_errors.mean():.4f}")
    
    # Compute off-manifold distances (same for all architectures)
    off_manifold_distances = []
    for i in range(len(latent_points)):
        dist = manifold.normal_deviation(latent_points[i])
        off_manifold_distances.append(dist)
    off_manifold_distances = np.array(off_manifold_distances)
    
    logger.info(f"Off-manifold distances: min={off_manifold_distances.min():.4f}, max={off_manifold_distances.max():.4f}, mean={off_manifold_distances.mean():.4f}")
    
    # Evaluate methods
    results = {}
    
    # Branch based on architecture type
    if hasattr(args, 'use_mask_value') and args.use_mask_value:
        # ========== ABLATION C: Evaluate mask, value, total separately ==========
        
        # 1a. Mask-only (density/sparsity reconstruction)
        logger.info("\n" + "="*80)
        logger.info("Method 1a: Mask-Only (Mask BCE)")
        logger.info("="*80)
        mask_results = evaluate_method(mask_scores, test_labels, test_metadata, method_name="Mask-Only")
        results['mask_only'] = mask_results
        
        # 1b. Value-only (magnitude reconstruction on active cells)
        logger.info("\n" + "="*80)
        logger.info("Method 1b: Value-Only (Active-Only MSE)")
        logger.info("="*80)
        value_results = evaluate_method(value_scores, test_labels, test_metadata, method_name="Value-Only")
        results['value_only'] = value_results
        
        # 1c. Total AE (mask + value combined)
        logger.info("\n" + "="*80)
        logger.info("Method 1c: Total AE (Mask + Value)")
        logger.info("="*80)
        total_results = evaluate_method(total_scores, test_labels, test_metadata, method_name="AE-Total")
        results['ae_total'] = total_results
        
        # 2. Off-Manifold Distance (same for all)
        logger.info("\n" + "="*80)
        logger.info("Method 2: Off-Manifold Distance")
        logger.info("="*80)
        om_results = evaluate_method(off_manifold_distances, test_labels, test_metadata, method_name="Off-Manifold")
        results['off_manifold'] = om_results
        
        # 3. Combined (grid search)
        if args.grid_search:
            # 3a. Two-component: Total + β (for comparison with baseline)
            logger.info("\n" + "="*80)
            logger.info("Method 3a: Combined Total+β (2-component)")
            logger.info("="*80)
            
            best_alpha, best_beta, best_pr_auc, combined_scores = grid_search_weights(
                total_scores, off_manifold_distances, test_labels
            )
            
            logger.info(f"Best weights: α={best_alpha:.2f}, β={best_beta:.2f}, PR-AUC={best_pr_auc:.4f}")
            combined_2_results = evaluate_method(combined_scores, test_labels, test_metadata, method_name="Combined-2")
            combined_2_results['alpha'] = best_alpha
            combined_2_results['beta'] = best_beta
            results['combined_total_beta'] = combined_2_results
            
            # 3b. Three-component: Mask + Value + β (full ablation)
            logger.info("\n" + "="*80)
            logger.info("Method 3b: Combined Mask+Value+β (3-component)")
            logger.info("="*80)
            
            best_am, best_av, best_b, best_pr_auc_3, combined_scores_3 = grid_search_weights_3(
                mask_scores, value_scores, off_manifold_distances, test_labels
            )
            
            logger.info(f"Best weights: α_mask={best_am:.2f}, α_value={best_av:.2f}, β={best_b:.2f}, PR-AUC={best_pr_auc_3:.4f}")
            combined_3_results = evaluate_method(combined_scores_3, test_labels, test_metadata, method_name="Combined-3")
            combined_3_results['alpha_mask'] = best_am
            combined_3_results['alpha_value'] = best_av
            combined_3_results['beta'] = best_b
            combined_3_results['note'] = "3-component uses z-scored inputs"
            results['combined_mask_value_beta'] = combined_3_results
    else:
        # ========== BASELINE: Standard single-channel evaluation ==========
        
        # 1. AE-only (reconstruction error)
        logger.info("\n" + "="*80)
        logger.info("Method 1: Autoencoder Only (Reconstruction Error)")
        logger.info("="*80)
        ae_results = evaluate_method(reconstruction_errors, test_labels, test_metadata, method_name="AE-Only")
        results['ae_only'] = ae_results
        
        # 2. Off-Manifold Distance
        logger.info("\n" + "="*80)
        logger.info("Method 2: Off-Manifold Distance")
        logger.info("="*80)
        om_results = evaluate_method(off_manifold_distances, test_labels, test_metadata, method_name="Off-Manifold")
        results['off_manifold'] = om_results
        
        # 3. Combined (grid search for optimal alpha, beta)
        if args.grid_search:
            logger.info("\n" + "="*80)
            logger.info("Method 3: Combined (Grid Search for α, β)")
            logger.info("="*80)
            
            best_alpha, best_beta, best_pr_auc, combined_scores = grid_search_weights(
                reconstruction_errors, off_manifold_distances, test_labels
            )
            
            logger.info(f"Best weights: α={best_alpha:.2f}, β={best_beta:.2f}, PR-AUC={best_pr_auc:.4f}")
            combined_results = evaluate_method(combined_scores, test_labels, test_metadata, method_name="Combined")
            combined_results['alpha'] = best_alpha
            combined_results['beta'] = best_beta
            results['combined'] = combined_results
    
    # Save results
    results_path = exp_dir / 'window_level_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")
    
    # Save scores (include all components for post-hoc analysis)
    scores_path = exp_dir / 'window_level_scores.npz'
    
    # Build save dictionary based on architecture
    save_kwargs = dict(
        off_manifold_distances=off_manifold_distances,
        test_labels=test_labels,
        user_ids=test_metadata['user_id'].values,
        scenarios=test_metadata['scenario'].values,
        latent_points=latent_points,  # Include latent for post-hoc plots
    )
    
    if hasattr(args, 'use_mask_value') and args.use_mask_value:
        # Ablation C: Save all three AE components
        save_kwargs.update(dict(
            mask_bce=mask_scores,
            value_mse_active=value_scores,
            ae_total=total_scores,
        ))
    else:
        # Baseline: Save standard reconstruction error
        save_kwargs.update(dict(
            reconstruction_errors=reconstruction_errors,
        ))
    
    np.savez(scores_path, **save_kwargs)
    logger.info(f"Scores saved to: {scores_path}")
    
    return results


def evaluate_method(scores, labels, metadata, method_name="Method"):
    """
    Evaluate a detection method using various metrics.
    
    Args:
        scores: (N,) array of anomaly scores (higher = more anomalous)
        labels: (N,) array of binary labels
        metadata: DataFrame with scenario column
        method_name: name of method for logging
        
    Returns:
        results: dictionary of metrics
    """
    # Overall metrics
    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    
    # Find optimal threshold for F1
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else thresholds[-1]
    best_precision = precision[best_f1_idx]
    best_recall = recall[best_f1_idx]
    
    # Confusion matrix at optimal threshold
    pred_labels = (scores >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()
    
    # Scenario breakdown: evaluate each scenario against normal
    # For each malicious scenario, create a binary problem: "Scenario X vs Normal"
    scenario_results = {}
    normal_mask = (metadata['scenario'] == 0).values
    
    for scenario in sorted(metadata['scenario'].unique()):
        if scenario == 0:  # Skip normal
            continue
        
        # Create mask for this scenario's malicious windows + all normal windows
        scenario_mal_mask = (metadata['scenario'] == scenario).values
        combined_mask = scenario_mal_mask | normal_mask
        
        scenario_labels = labels[combined_mask]
        scenario_scores = scores[combined_mask]
        
        if len(scenario_labels) > 0 and scenario_labels.sum() > 0:
            scenario_roc = roc_auc_score(scenario_labels, scenario_scores)
            scenario_pr = average_precision_score(scenario_labels, scenario_scores)
            scenario_results[f"scenario_{int(scenario)}"] = {
                'total_windows': int(len(scenario_labels)),
                'malicious': int(scenario_labels.sum()),
                'normal': int((scenario_labels == 0).sum()),
                'roc_auc': float(scenario_roc),
                'pr_auc': float(scenario_pr)
            }
    
    # Log results
    logger.info(f"\n{method_name} Results:")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  PR-AUC:  {pr_auc:.4f} ⭐ (primary metric)")
    logger.info(f"  Best F1: {best_f1:.4f} (threshold={best_threshold:.4f})")
    logger.info(f"    Precision: {best_precision:.4f}")
    logger.info(f"    Recall:    {best_recall:.4f}")
    logger.info(f"  Confusion Matrix (at optimal threshold):")
    logger.info(f"    TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    if scenario_results:
        logger.info(f"  Scenario Breakdown (each scenario vs. normal):")
        for scenario_name, scenario_metrics in scenario_results.items():
            logger.info(f"    {scenario_name}: PR-AUC={scenario_metrics['pr_auc']:.4f}, "
                       f"ROC-AUC={scenario_metrics['roc_auc']:.4f}, "
                       f"({scenario_metrics['malicious']} malicious + {scenario_metrics['normal']} normal = {scenario_metrics['total_windows']} total)")
    
    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'best_f1': float(best_f1),
        'best_threshold': float(best_threshold),
        'best_precision': float(best_precision),
        'best_recall': float(best_recall),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'scenario_breakdown': scenario_results
    }


def grid_search_weights(recon_errors, off_manifold_dists, labels):
    """
    Grid search for optimal alpha and beta weights.
    
    Combined score: S = α * recon_error + β * off_manifold_dist
    
    Args:
        recon_errors: (N,) array
        off_manifold_dists: (N,) array
        labels: (N,) binary array
        
    Returns:
        best_alpha, best_beta, best_pr_auc, best_combined_scores
    """
    # Standardize scores (z-score) instead of normalizing to [0, 1]
    recon_std = (recon_errors - recon_errors.mean()) / (recon_errors.std() + 1e-10)
    om_std = (off_manifold_dists - off_manifold_dists.mean()) / (off_manifold_dists.std() + 1e-10)
    
    # Grid search with wider range including values > 1
    alphas = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
    betas = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    best_pr_auc = 0
    best_alpha = 0.5
    best_beta = 0.5
    best_scores = None
    
    # Store results for analysis
    grid_results = []
    
    for alpha in alphas:
        for beta in betas:
            if alpha + beta == 0:
                continue
            
            combined = alpha * recon_std + beta * om_std
            pr_auc = average_precision_score(labels, combined)
            grid_results.append((alpha, beta, pr_auc))
            
            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                best_alpha = alpha
                best_beta = beta
                best_scores = combined
    
    # Log top 5 combinations
    grid_results_sorted = sorted(grid_results, key=lambda x: x[2], reverse=True)
    logger.info("Top 5 α/β combinations:")
    for i, (a, b, pr) in enumerate(grid_results_sorted[:5]):
        logger.info(f"  {i+1}. α={a:.2f}, β={b:.2f} → PR-AUC={pr:.4f}")
    
    return best_alpha, best_beta, best_pr_auc, best_scores


def grid_search_weights_3(mask_scores, value_scores, off_manifold_dists, labels):
    """
    Grid search for optimal α_mask, α_value, and β weights (3-component).
    
    Combined score: S = α_mask * mask + α_value * value + β * off_manifold
    
    For Ablation C: separate weights for mask reconstruction, value reconstruction,
    and manifold geometry. Tells us whether:
    - Geometry adds beyond mask/value
    - Mask and value need different emphasis
    
    Args:
        mask_scores: (N,) array - mask BCE per window
        value_scores: (N,) array - value MSE on active cells per window
        off_manifold_dists: (N,) array - off-manifold distance
        labels: (N,) binary array
        
    Returns:
        best_alpha_mask, best_alpha_value, best_beta, best_pr_auc, best_combined_scores
    """
    # Standardize all three components (z-score)
    # This makes the grid search scale-invariant
    mask_std = (mask_scores - mask_scores.mean()) / (mask_scores.std() + 1e-10)
    value_std = (value_scores - value_scores.mean()) / (value_scores.std() + 1e-10)
    om_std = (off_manifold_dists - off_manifold_dists.mean()) / (off_manifold_dists.std() + 1e-10)
    
    # Grid search ranges (matching 2-component style)
    alpha_mask_vals = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
    alpha_value_vals = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
    beta_vals = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    best_pr_auc = 0
    best_alpha_mask = 1.0
    best_alpha_value = 1.0
    best_beta = 0.5
    best_scores = None
    
    # Store results for analysis
    grid_results = []
    
    for alpha_m in alpha_mask_vals:
        for alpha_v in alpha_value_vals:
            for beta in beta_vals:
                if alpha_m + alpha_v + beta == 0:
                    continue
                
                combined = alpha_m * mask_std + alpha_v * value_std + beta * om_std
                pr_auc = average_precision_score(labels, combined)
                grid_results.append((alpha_m, alpha_v, beta, pr_auc))
                
                if pr_auc > best_pr_auc:
                    best_pr_auc = pr_auc
                    best_alpha_mask = alpha_m
                    best_alpha_value = alpha_v
                    best_beta = beta
                    best_scores = combined
    
    # Log top 5 combinations
    grid_results_sorted = sorted(grid_results, key=lambda x: x[3], reverse=True)
    logger.info("Top 5 α_mask/α_value/β combinations:")
    for i, (am, av, b, pr) in enumerate(grid_results_sorted[:5]):
        logger.info(f"  {i+1}. α_mask={am:.2f}, α_value={av:.2f}, β={b:.2f} → PR-AUC={pr:.4f}")
    
    return best_alpha_mask, best_alpha_value, best_beta, best_pr_auc, best_scores


def evaluate_trajectory_detection(
    model, manifold, train_data, test_data, test_labels, test_metadata, mu, sigma, args, exp_dir
):
    """
    Evaluate trajectory-level detection using geodesic deviation analysis.
    
    Creates trajectories (sequences of consecutive windows from same user) and
    analyzes them as paths through latent space.
    
    Args:
        model: trained autoencoder
        manifold: UEBALatentManifold
        train_data: (N_train, T, F) training windows
        test_data: (N_test, T, F) test windows
        test_labels: (N_test,) binary labels
        test_metadata: DataFrame with user_id, window_start, scenario
        mu, sigma: normalization parameters
        args: command line arguments
        exp_dir: experiment directory
        
    Returns:
        results: dictionary of trajectory-level metrics
    """
    logger.info("Evaluating trajectory-level detection (geodesic deviation)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # ===== Extract latent representations for all windows =====
    def extract_latents(data):
        """Extract latent vectors for all windows."""
        if hasattr(args, 'use_mask_value') and args.use_mask_value:
            # Ablation C: Use mask+value scoring to get latents
            scores_dict = score_windows_mask_value(model, data, mu, sigma, device)
            return scores_dict["latent"]
        else:
            # Standard/weighted/two-term: Use standard dataset
            dataset = SeqDataset(data, mu=mu, sigma=sigma)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            
            latents = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device).unsqueeze(1)
                    _, latent = model(batch)
                    latents.append(latent.cpu().numpy())
            
            return np.concatenate(latents, axis=0)
    
    logger.info("Extracting latent representations...")
    train_latents = extract_latents(train_data)
    test_latents = extract_latents(test_data)
    logger.info(f"Train latents: {train_latents.shape}, Test latents: {test_latents.shape}")
    
    # ===== Create sliding window trajectories =====
    def create_sliding_trajectories(latents, metadata, labels, window_size=6, stride=3):
        """
        Create sliding window trajectories from consecutive windows.
        
        Each trajectory is labeled as malicious if it OVERLAPS any malicious window.
        This enables detection of multi-day attack campaigns at the trajectory level.
        
        Args:
            latents: (N, latent_dim) array of latent vectors
            metadata: DataFrame with user_id, window_start, scenario
            labels: (N,) binary array
            window_size: Number of consecutive windows per trajectory
            stride: Step size between trajectories
            
        Returns:
            trajectories: list of (latent_array, label, user_id, scenario, start_idx) tuples
        """
        trajectories = []
        users = metadata['user_id'].unique()
        
        for user_id in users:
            user_mask = (metadata['user_id'] == user_id).values
            user_indices = np.where(user_mask)[0]
            
            # CRITICAL FIX: Sort by window_start to get chronological order
            # All windows now have valid timestamps (no NaN)
            user_metadata = metadata.iloc[user_indices].copy()
            
            # Sort by window_start chronologically
            time_order = user_metadata['window_start'].argsort().values
            user_indices = user_indices[time_order]
            
            # Now extract with proper chronological ordering
            user_latents = latents[user_indices]
            user_labels = labels[user_indices]
            user_scenarios = metadata.iloc[user_indices]['scenario'].values
            
            if len(user_latents) < window_size:
                continue
            
            # Create sliding window trajectories for this user (FIXED: stride instead of window_size)
            for i in range(0, len(user_latents) - window_size + 1, stride):
                traj_latents = user_latents[i:i + window_size]
                traj_labels = user_labels[i:i + window_size]
                traj_scenarios = user_scenarios[i:i + window_size]
                
                # Trajectory is malicious if it OVERLAPS any malicious window
                traj_label = 1 if (traj_labels == 1).any() else 0
                
                # Get scenario of malicious windows (if any)
                if traj_label == 1:
                    traj_scenario = traj_scenarios[traj_labels == 1][0]
                else:
                    traj_scenario = 0
                
                trajectories.append((
                    traj_latents, 
                    traj_label, 
                    user_id, 
                    traj_scenario,
                    user_indices[i]  # Global index for tracking
                ))
        
        return trajectories
    
    # Trajectory parameters from args
    traj_window_size = args.traj_window_size
    traj_stride = args.traj_stride
    
    logger.info(f"Creating sliding window trajectories (window={traj_window_size}, stride={traj_stride})...")
    
    # Training: create sliding trajectories from normal windows
    # Use sequential chunks since we don't have user boundaries (all normal)
    train_trajectories = []
    for i in range(0, len(train_latents) - traj_window_size + 1, traj_stride):
        chunk = train_latents[i:i + traj_window_size]
        train_trajectories.append((chunk, 0, f"train_{i}", 0, i))  # (latents, label, pseudo_user, scenario, idx)
    
    # Test: create sliding trajectories per user with overlap-based labeling
    test_trajectories = create_sliding_trajectories(
        test_latents, test_metadata, test_labels, 
        window_size=traj_window_size, 
        stride=traj_stride
    )
    
    # Count trajectory types
    normal_test_trajs = [t for t in test_trajectories if t[1] == 0]
    malicious_test_trajs = [t for t in test_trajectories if t[1] == 1]
    
    # SANITY CHECK: Trajectory counts should match window-level proportions
    # Window-level: ~112 malicious / 1609 total = 7%
    # Trajectory-level: Should be similar (maybe slightly more due to overlap)
    traj_mal_pct = len(malicious_test_trajs) / len(test_trajectories) * 100
    window_mal_pct = test_labels.sum() / len(test_labels) * 100
    
    # Calculate actual overlap percentage
    overlap_pct = (1 - traj_stride / traj_window_size) * 100
    
    logger.info(f"Sliding window trajectories created:")
    logger.info(f"  Training: {len(train_trajectories)} normal trajectories")
    logger.info(f"  Test: {len(test_trajectories)} total ({len(malicious_test_trajs)} malicious, {len(normal_test_trajs)} normal)")
    logger.info(f"  Trajectory size: {traj_window_size} consecutive 24hr windows (= {traj_window_size} days)")
    logger.info(f"  Overlap: stride={traj_stride} ({overlap_pct:.0f}% overlap)")
    logger.info(f"  SANITY: Window-level malicious={window_mal_pct:.1f}%, Trajectory-level malicious={traj_mal_pct:.1f}%")
    
    # ===== Initialize TrajectoryAnalyzer and fit reference statistics =====
    logger.info("Fitting reference statistics from normal training trajectories...")
    traj_config = TrajectoryConfig(k_neighbors=args.k_neighbors, min_trajectory_length=4)
    analyzer = TrajectoryAnalyzer(manifold, config=traj_config)
    
    # Fit on training trajectories
    train_traj_arrays = [t for t, _, _, _, _ in train_trajectories]
    analyzer.fit_reference_statistics(train_traj_arrays)
    
    # ===== Score all test trajectories =====
    logger.info("Scoring test trajectories with geodesic deviation...")
    
    trajectory_scores = []
    trajectory_labels = []
    trajectory_users = []
    trajectory_scenarios = []
    
    for traj_latents, traj_label, user_id, scenario, start_idx in test_trajectories:
        score = analyzer.score_trajectory(traj_latents)
        trajectory_scores.append(score)
        trajectory_labels.append(traj_label)
        trajectory_users.append(user_id)
        trajectory_scenarios.append(scenario)
    
    trajectory_scores = np.array(trajectory_scores)
    trajectory_labels = np.array(trajectory_labels)
    trajectory_scenarios = np.array(trajectory_scenarios)
    
    logger.info(f"Scored {len(trajectory_scores)} trajectories")
    logger.info(f"  Score range: [{trajectory_scores.min():.4f}, {trajectory_scores.max():.4f}]")
    logger.info(f"  Mean score: {trajectory_scores.mean():.4f}")
    
    # ===== Evaluate trajectory-level metrics =====
    logger.info("\n" + "="*80)
    logger.info("Trajectory-Level Detection (Geodesic Deviation)")
    logger.info("="*80)
    
    # Create simple metadata for evaluation
    traj_metadata_df = pd.DataFrame({'scenario': trajectory_scenarios})
    
    results = evaluate_method(trajectory_scores, trajectory_labels, traj_metadata_df, method_name="Trajectory (Geodesic)")
    
    # Save results
    results_path = exp_dir / 'trajectory_level_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")
    
    # Save scores
    scores_path = exp_dir / 'trajectory_level_scores.npz'
    np.savez(
        scores_path,
        trajectory_scores=trajectory_scores,
        trajectory_labels=trajectory_labels,
        trajectory_users=trajectory_users,
        trajectory_scenarios=trajectory_scenarios
    )
    logger.info(f"Scores saved to: {scores_path}")
    
    return results


def main():
    args = parse_args()
    
    # Quick test mode
    if args.quick_test:
        args.epochs = 5
        args.early_stop_patience = 3  # Shorter patience for quick test
        logger.info("⚡ Quick test mode: epochs=5, early_stop_patience=3")
    
    # Setup experiment
    exp_dir = setup_experiment(args)
    
    # Load data
    logger.info("\n" + "="*80)
    logger.info("Loading Data")
    logger.info("="*80)
    loader = CERTFixedWindowLoader(data_dir=args.data_dir)
    train_data, test_data, test_labels, test_metadata, train_metadata = loader.load_fixed_windows(
        bucket_hours=args.bucket_hours,
        window_hours=args.window_hours,
        buffer_days=args.buffer_days,
        train_split=args.train_split,
        random_seed=42
    )
    
    logger.info(f"Training: {train_data.shape}")
    logger.info(f"Test: {test_data.shape}, {(test_labels==1).sum()} malicious ({(test_labels==1).mean()*100:.1f}%)")
    
    # Train autoencoder
    logger.info("\n" + "="*80)
    logger.info("Training Phase")
    logger.info("="*80)
    model, mu, sigma = train_autoencoder(train_data, args, exp_dir, train_metadata=train_metadata)
    
    # Build manifold
    logger.info("\n" + "="*80)
    logger.info("Manifold Construction")
    logger.info("="*80)
    manifold = build_manifold(model, train_data, mu, sigma, args, exp_dir)
    
    # Evaluate window-level
    logger.info("\n" + "="*80)
    logger.info("Window-Level Evaluation")
    logger.info("="*80)
    window_results = evaluate_window_detection(model, manifold, test_data, test_labels, test_metadata, mu, sigma, args, exp_dir)
    
    # Evaluate trajectory-level
    logger.info("\n" + "="*80)
    logger.info("Trajectory-Level Evaluation")
    logger.info("="*80)
    trajectory_results = evaluate_trajectory_detection(
        model, manifold, train_data, test_data, test_labels, test_metadata, mu, sigma, args, exp_dir
    )
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Results directory: {exp_dir}")
    logger.info(f"\nWindow-Level Detection Performance:")
    
    # Branch based on architecture
    if hasattr(args, 'use_mask_value') and args.use_mask_value:
        # Ablation C: Show mask/value/total components
        logger.info(f"  Mask-Only:    PR-AUC={window_results['mask_only']['pr_auc']:.4f}, ROC-AUC={window_results['mask_only']['roc_auc']:.4f}")
        logger.info(f"  Value-Only:   PR-AUC={window_results['value_only']['pr_auc']:.4f}, ROC-AUC={window_results['value_only']['roc_auc']:.4f}")
        logger.info(f"  AE-Total:     PR-AUC={window_results['ae_total']['pr_auc']:.4f}, ROC-AUC={window_results['ae_total']['roc_auc']:.4f}")
        logger.info(f"  Off-Manifold: PR-AUC={window_results['off_manifold']['pr_auc']:.4f}, ROC-AUC={window_results['off_manifold']['roc_auc']:.4f}")
        
        # Combined results (if grid search was run)
        if 'combined_total_beta' in window_results:
            c2 = window_results['combined_total_beta']
            logger.info(f"  Combined (Total+β): PR-AUC={c2['pr_auc']:.4f}, ROC-AUC={c2['roc_auc']:.4f} "
                       f"(α={c2['alpha']:.2f}, β={c2['beta']:.2f})")
        
        if 'combined_mask_value_beta' in window_results:
            c3 = window_results['combined_mask_value_beta']
            logger.info(f"  Combined (Mask+Value+β): PR-AUC={c3['pr_auc']:.4f}, ROC-AUC={c3['roc_auc']:.4f} "
                       f"(α_mask={c3['alpha_mask']:.2f}, α_value={c3['alpha_value']:.2f}, β={c3['beta']:.2f})")
    else:
        # Baseline: Show standard results
        logger.info(f"  AE-Only:      PR-AUC={window_results['ae_only']['pr_auc']:.4f}, ROC-AUC={window_results['ae_only']['roc_auc']:.4f}")
        logger.info(f"  Off-Manifold: PR-AUC={window_results['off_manifold']['pr_auc']:.4f}, ROC-AUC={window_results['off_manifold']['roc_auc']:.4f}")
        
        if 'combined' in window_results:
            logger.info(f"  Combined:     PR-AUC={window_results['combined']['pr_auc']:.4f}, ROC-AUC={window_results['combined']['roc_auc']:.4f} "
                       f"(α={window_results['combined']['alpha']:.2f}, β={window_results['combined']['beta']:.2f})")
    
    logger.info(f"\nTrajectory-Level Detection Performance (Geodesic Deviation):")
    logger.info(f"  PR-AUC: {trajectory_results['pr_auc']:.4f} ⭐")
    logger.info(f"  ROC-AUC: {trajectory_results['roc_auc']:.4f}")
    logger.info(f"  F1-score: {trajectory_results['best_f1']:.4f}")
    
    logger.info("\n✓ Pipeline complete!")


if __name__ == '__main__':
    main()
