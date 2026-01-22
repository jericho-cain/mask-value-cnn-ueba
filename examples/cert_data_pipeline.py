"""
Example: CERT Insider Threat Dataset Pipeline

This script demonstrates how to:
1. Load CERT r4.2 dataset
2. Process it into CNN-compatible format
3. Train the autoencoder on normal behavior
4. Evaluate manifold-based anomaly detection

Usage:
    # First, download CERT data manually (see instructions)
    python examples/cert_data_pipeline.py --data-dir data/cert/r4.2 --sample 50

For local testing with small sample:
    python examples/cert_data_pipeline.py --data-dir data/cert/r4.2 --sample 50

For full dataset (on cluster):
    python examples/cert_data_pipeline.py --data-dir data/cert/r4.2
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from manifold_ueba.etl.cert import CERTDataLoader, CERT_FEATURES, download_cert_dataset
from manifold_ueba.cnn_model import UEBACNNAutoencoder, prepare_ueba_sequences_for_cnn
from manifold_ueba.data import compute_stats, SeqDataset
from manifold_ueba.latent_manifold import UEBALatentManifold, UEBAManifoldConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="CERT data pipeline for UEBA manifold learning")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/cert/r4.2",
        help="Path to CERT dataset directory"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of users to sample (for local testing). Default: all users"
    )
    parser.add_argument(
        "--bucket-hours",
        type=float,
        default=1.0,
        help="Hours per time bucket. Use 0.25 for 15-min buckets. Default: 1.0"
    )
    parser.add_argument(
        "--min-attack-hours",
        type=float,
        default=None,
        help="Minimum attack duration in hours to include. Filters out shorter attacks."
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=24,
        help="Number of time buckets per sequence. Default: 24"
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=32,
        help="Latent space dimension. Default: 32"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs. Default: 50"
    )
    parser.add_argument(
        "--download-instructions",
        action="store_true",
        help="Show download instructions and exit"
    )
    parser.add_argument(
        "--save-processed",
        type=str,
        default=None,
        help="Save processed data to this .npz file"
    )
    parser.add_argument(
        "--load-processed",
        type=str,
        default=None,
        help="Load processed data from this .npz file (skips CSV loading)"
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run grid search over alpha/beta values"
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Save trained model to this .pt file"
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Load trained model from this .pt file (skips training)"
    )
    parser.add_argument(
        "--save-manifold",
        type=str,
        default=None,
        help="Save manifold (latents + config) to this .npz file"
    )
    parser.add_argument(
        "--load-manifold",
        type=str,
        default=None,
        help="Load manifold from this .npz file (skips manifold construction)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Show download instructions if requested
    if args.download_instructions:
        download_cert_dataset(args.data_dir)
        return
    
    data_dir = Path(args.data_dir)
    
    # Check if data exists
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Run with --download-instructions to see how to get the data")
        download_cert_dataset(data_dir)
        return
    
    # Load data - either from processed .npz or from raw CSVs
    logger.info("=" * 60)
    logger.info("CERT Data Loading")
    logger.info("=" * 60)
    
    if args.load_processed:
        # Load from saved .npz file
        logger.info(f"Loading processed data from {args.load_processed}")
        loaded = np.load(args.load_processed)
        train_data = loaded['train_data']
        test_data = loaded['test_data']
        test_labels = loaded['test_labels']
        logger.info("Loaded successfully!")
    else:
        # Load from raw CSVs
        loader = CERTDataLoader(data_dir)
        
        try:
            train_data, test_data, test_labels = loader.load_splits(
                n_users_sample=args.sample,
                bucket_hours=args.bucket_hours,
                sequence_length=args.sequence_length,
                min_attack_hours=args.min_attack_hours
            )
        except FileNotFoundError as e:
            logger.error(str(e))
            return
        
        # Save if requested
        if args.save_processed:
            logger.info(f"Saving processed data to {args.save_processed}")
            np.savez(
                args.save_processed,
                train_data=train_data,
                test_data=test_data,
                test_labels=test_labels
            )
            logger.info("Saved successfully!")
    
    logger.info(f"Train data shape: {train_data.shape}")
    logger.info(f"Test data shape: {test_data.shape}")
    logger.info(f"Test labels: {test_labels.sum()} malicious, {(test_labels == 0).sum()} normal")
    
    if len(train_data) == 0:
        logger.error("No training data generated. Check data directory contents.")
        return
    
    # Compute normalization stats from training data
    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing")
    logger.info("=" * 60)
    
    # Flatten for stats computation
    train_flat = train_data.reshape(-1, train_data.shape[-1])
    mu, sigma = compute_stats(train_flat)
    
    logger.info(f"Feature means: {mu[:5]}... (first 5)")
    logger.info(f"Feature stds: {sigma[:5]}... (first 5)")
    
    # Create datasets
    train_sequences = [train_data[i] for i in range(len(train_data))]
    test_sequences = [test_data[i] for i in range(len(test_data))]
    
    train_dataset = SeqDataset(train_sequences, mu, sigma)
    test_dataset = SeqDataset(test_sequences, mu, sigma)
    
    # Prepare for CNN (add channel dimension, normalize)
    n_features = train_data.shape[-1]
    sequence_length = train_data.shape[1]
    
    logger.info(f"Sequence length (T): {sequence_length}")
    logger.info(f"Number of features (F): {n_features}")
    logger.info(f"Feature names: {CERT_FEATURES}")
    
    # Initialize model
    logger.info("\n" + "=" * 60)
    logger.info("Model Initialization")
    logger.info("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = UEBACNNAutoencoder(
        time_steps=sequence_length,
        n_features=n_features,
        latent_dim=args.latent_dim
    )
    
    # Load or train model
    if args.load_model:
        logger.info(f"Loading model from {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        logger.info("Model loaded successfully!")
    else:
        model_info = model.get_model_info()
        logger.info(f"Model architecture: {model_info['architecture']}")
        logger.info(f"Total parameters: {model_info['total_parameters']:,}")
        logger.info(f"Model size: {model_info['model_size_mb']:.2f} MB")
        
        # Training
        logger.info("\n" + "=" * 60)
        logger.info("Training")
        logger.info("=" * 60)
        
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = torch.nn.MSELoss()
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            
            for batch in train_loader:
                # Prepare batch for CNN
                batch = batch.unsqueeze(1).to(device)  # Add channel dim
                
                optimizer.zero_grad()
                reconstructed, latent = model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.6f}")
        
        # Save model if requested
        if args.save_model:
            logger.info(f"Saving model to {args.save_model}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'latent_dim': args.latent_dim,
                'time_steps': sequence_length,
                'n_features': n_features,
                'mu': mu,
                'sigma': sigma,
            }, args.save_model)
            logger.info("Model saved successfully!")
    
    # Extract latent representations for manifold construction
    logger.info("\n" + "=" * 60)
    logger.info("Manifold Construction")
    logger.info("=" * 60)
    
    if args.load_manifold:
        logger.info(f"Loading manifold from {args.load_manifold}")
        manifold_data = np.load(args.load_manifold)
        train_latents = manifold_data['train_latents']
        k_neighbors = int(manifold_data['k_neighbors'])
        manifold_config = UEBAManifoldConfig(k_neighbors=k_neighbors)
        manifold = UEBALatentManifold(train_latents, manifold_config)
        logger.info(f"Manifold loaded: {train_latents.shape[0]} points, k={k_neighbors}")
    else:
        model.eval()
        train_latents = []
        
        with torch.no_grad():
            for batch in DataLoader(train_dataset, batch_size=64, shuffle=False):
                batch = batch.unsqueeze(1).to(device)
                latent = model.encode(batch)
                train_latents.append(latent.cpu().numpy())
        
        train_latents = np.vstack(train_latents)
        logger.info(f"Training latent representations: {train_latents.shape}")
        
        # Build manifold
        k_neighbors = min(32, len(train_latents) - 1)
        manifold_config = UEBAManifoldConfig(k_neighbors=k_neighbors)
        manifold = UEBALatentManifold(train_latents, manifold_config)
        
        # Save manifold if requested
        if args.save_manifold:
            logger.info(f"Saving manifold to {args.save_manifold}")
            np.savez(
                args.save_manifold,
                train_latents=train_latents,
                k_neighbors=k_neighbors,
            )
            logger.info("Manifold saved successfully!")
    
    quality = manifold.validate_manifold_quality()
    logger.info(f"Manifold coherence: {quality['manifold_coherence']:.4f}")
    logger.info(f"Mean neighbor distance: {quality['avg_neighbor_distance']:.4f}")
    
    # Evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation")
    logger.info("=" * 60)
    
    # First, compute reconstruction errors and manifold distances for all test samples
    test_recon_errors = []
    test_manifold_dists = []
    test_latents = []
    
    model.eval()
    with torch.no_grad():
        for i, seq in enumerate(test_dataset):
            seq_tensor = seq.unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dims
            
            reconstructed, latent = model(seq_tensor)
            
            # Reconstruction error
            recon_error = ((seq_tensor - reconstructed) ** 2).mean().item()
            
            # Off-manifold distance
            latent_np = latent.cpu().numpy()
            manifold_dist = manifold.normal_deviation(latent_np[0])
            
            test_recon_errors.append(recon_error)
            test_manifold_dists.append(manifold_dist)
            test_latents.append(latent_np[0])
    
    test_recon_errors = np.array(test_recon_errors)
    test_manifold_dists = np.array(test_manifold_dists)
    
    # Grid search or single evaluation
    if args.grid_search:
        logger.info("\nRunning grid search over alpha/beta...")
        logger.info("-" * 60)
        
        # Grid search parameters
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        betas = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
        
        results = []
        for alpha in alphas:
            for beta in betas:
                if alpha == 0 and beta == 0:
                    continue  # Skip trivial case
                
                scores = alpha * test_recon_errors + beta * test_manifold_dists
                roc_auc = roc_auc_score(test_labels, scores)
                pr_auc = average_precision_score(test_labels, scores)
                
                results.append({
                    'alpha': alpha,
                    'beta': beta,
                    'roc_auc': roc_auc,
                    'pr_auc': pr_auc
                })
        
        # Sort by PR-AUC (most relevant for imbalanced data)
        results.sort(key=lambda x: x['pr_auc'], reverse=True)
        
        logger.info("\nTop 10 configurations by PR-AUC:")
        logger.info(f"{'Alpha':<8} {'Beta':<8} {'ROC-AUC':<10} {'PR-AUC':<10}")
        logger.info("-" * 40)
        for r in results[:10]:
            logger.info(f"{r['alpha']:<8.2f} {r['beta']:<8.2f} {r['roc_auc']:<10.4f} {r['pr_auc']:<10.4f}")
        
        # Use best config for detailed evaluation
        best = results[0]
        alpha, beta = best['alpha'], best['beta']
        logger.info(f"\nBest config: alpha={alpha}, beta={beta}")
    else:
        # Default values
        alpha = 0.5
        beta = 2.0
    
    # Compute scores with chosen alpha/beta
    test_scores_combined = alpha * test_recon_errors + beta * test_manifold_dists
    test_scores_ae_only = test_recon_errors  # beta=0 case
    
    # Compute separation metrics
    normal_mask = test_labels == 0
    malicious_mask = test_labels == 1
    
    if malicious_mask.sum() > 0 and normal_mask.sum() > 0:
        # AE-only separation
        ae_normal_mean = test_scores_ae_only[normal_mask].mean()
        ae_malicious_mean = test_scores_ae_only[malicious_mask].mean()
        ae_normal_std = test_scores_ae_only[normal_mask].std()
        ae_malicious_std = test_scores_ae_only[malicious_mask].std()
        ae_separation = (ae_malicious_mean - ae_normal_mean) / (ae_normal_std + ae_malicious_std + 1e-8)
        
        # Combined separation
        comb_normal_mean = test_scores_combined[normal_mask].mean()
        comb_malicious_mean = test_scores_combined[malicious_mask].mean()
        comb_normal_std = test_scores_combined[normal_mask].std()
        comb_malicious_std = test_scores_combined[malicious_mask].std()
        comb_separation = (comb_malicious_mean - comb_normal_mean) / (comb_normal_std + comb_malicious_std + 1e-8)
        
        logger.info(f"\nAE-only (beta=0):")
        logger.info(f"  Normal mean: {ae_normal_mean:.4f}, Malicious mean: {ae_malicious_mean:.4f}")
        logger.info(f"  Separation score: {ae_separation:.4f}")
        
        logger.info(f"\nCombined (alpha={alpha}, beta={beta}):")
        logger.info(f"  Normal mean: {comb_normal_mean:.4f}, Malicious mean: {comb_malicious_mean:.4f}")
        logger.info(f"  Separation score: {comb_separation:.4f}")
        
        improvement = (comb_separation - ae_separation) / (ae_separation + 1e-8) * 100
        logger.info(f"\nImprovement from manifold geometry: {improvement:.1f}%")
        
        # ROC-AUC and PR-AUC metrics
        logger.info("\n" + "-" * 40)
        logger.info("Classification Metrics")
        logger.info("-" * 40)
        
        # AE-only metrics
        ae_roc_auc = roc_auc_score(test_labels, test_scores_ae_only)
        ae_pr_auc = average_precision_score(test_labels, test_scores_ae_only)
        
        # Combined metrics
        comb_roc_auc = roc_auc_score(test_labels, test_scores_combined)
        comb_pr_auc = average_precision_score(test_labels, test_scores_combined)
        
        logger.info(f"\nAE-only (beta=0):")
        logger.info(f"  ROC-AUC: {ae_roc_auc:.4f}")
        logger.info(f"  PR-AUC:  {ae_pr_auc:.4f}")
        
        logger.info(f"\nCombined (alpha={alpha}, beta={beta}):")
        logger.info(f"  ROC-AUC: {comb_roc_auc:.4f}")
        logger.info(f"  PR-AUC:  {comb_pr_auc:.4f}")
        
        roc_improvement = (comb_roc_auc - ae_roc_auc) / ae_roc_auc * 100
        pr_improvement = (comb_pr_auc - ae_pr_auc) / ae_pr_auc * 100
        logger.info(f"\nROC-AUC improvement: {roc_improvement:.1f}%")
        logger.info(f"PR-AUC improvement:  {pr_improvement:.1f}%")
        
        # Precision at specific recall levels
        logger.info("\n" + "-" * 40)
        logger.info("Precision at Fixed Recall Levels")
        logger.info("-" * 40)
        
        for target_recall in [0.90, 0.95, 0.99]:
            # AE-only
            ae_precision, ae_recall, _ = precision_recall_curve(test_labels, test_scores_ae_only)
            ae_idx = np.argmin(np.abs(ae_recall - target_recall))
            ae_prec_at_recall = ae_precision[ae_idx]
            
            # Combined
            comb_precision, comb_recall, _ = precision_recall_curve(test_labels, test_scores_combined)
            comb_idx = np.argmin(np.abs(comb_recall - target_recall))
            comb_prec_at_recall = comb_precision[comb_idx]
            
            logger.info(f"At {int(target_recall*100)}% recall: AE-only precision={ae_prec_at_recall:.2%}, Combined precision={comb_prec_at_recall:.2%}")
        
        # False positive rate at specific recall levels
        logger.info("\n" + "-" * 40)
        logger.info("False Positive Rate at Fixed Recall (TPR)")
        logger.info("-" * 40)
        
        for target_tpr in [0.90, 0.95, 0.99]:
            # AE-only
            ae_fpr, ae_tpr, _ = roc_curve(test_labels, test_scores_ae_only)
            ae_idx = np.argmin(np.abs(ae_tpr - target_tpr))
            ae_fpr_at_tpr = ae_fpr[ae_idx]
            
            # Combined
            comb_fpr, comb_tpr, _ = roc_curve(test_labels, test_scores_combined)
            comb_idx = np.argmin(np.abs(comb_tpr - target_tpr))
            comb_fpr_at_tpr = comb_fpr[comb_idx]
            
            logger.info(f"At {int(target_tpr*100)}% detection: AE-only FPR={ae_fpr_at_tpr:.2%}, Combined FPR={comb_fpr_at_tpr:.2%}")
        
    else:
        logger.warning("Insufficient malicious or normal samples for evaluation")
    
    logger.info("\n" + "=" * 60)
    logger.info("Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
