"""
Trajectory Analysis on CERT Dataset

This script tests geodesic deviation (trajectory-based anomaly detection)
on CERT insider threat data with appropriate attack selection based on
attack duration vs. temporal resolution.

Usage:
    # First, train and save model + manifold
    python examples/cert_data_pipeline.py \
        --data-dir data/cert/r4.2 --sample 50 --epochs 20 \
        --save-processed data/cert_processed_50users.npz \
        --save-model data/cert_model.pt \
        --save-manifold data/cert_manifold.npz

    # Then run trajectory analysis
    python examples/trajectory_analysis.py \
        --load-processed data/cert_processed_50users.npz \
        --load-model data/cert_model.pt \
        --load-manifold data/cert_manifold.npz
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from manifold_ueba.cnn_model import UEBACNNAutoencoder
from manifold_ueba.data import compute_stats, SeqDataset
from manifold_ueba.latent_manifold import UEBALatentManifold, UEBAManifoldConfig
from manifold_ueba.trajectory import TrajectoryAnalyzer, TrajectoryConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_attack_durations(data_dir: Path) -> pd.DataFrame:
    """Load and compute attack durations from CERT answers."""
    answers_path = data_dir / "answers" / "insiders.csv"
    if not answers_path.exists():
        logger.warning(f"Answers file not found: {answers_path}")
        return pd.DataFrame()
    
    answers = pd.read_csv(answers_path)
    r42 = answers[answers["dataset"].astype(str).str.contains("4.2")].copy()
    
    r42["start_dt"] = pd.to_datetime(r42["start"])
    r42["end_dt"] = pd.to_datetime(r42["end"])
    r42["duration_hours"] = (r42["end_dt"] - r42["start_dt"]).dt.total_seconds() / 3600
    
    return r42[["user", "scenario", "duration_hours", "start_dt", "end_dt"]]


def parse_args():
    parser = argparse.ArgumentParser(description="Trajectory analysis on CERT data")
    parser.add_argument("--load-processed", type=str, required=True,
                        help="Path to processed data .npz file")
    parser.add_argument("--load-model", type=str, required=True,
                        help="Path to trained model .pt file")
    parser.add_argument("--load-manifold", type=str, required=True,
                        help="Path to manifold .npz file")
    parser.add_argument("--data-dir", type=str, default="data/cert/r4.2",
                        help="Path to CERT data (for attack duration info)")
    parser.add_argument("--window-size", type=int, default=6,
                        help="Number of consecutive latent points per trajectory")
    parser.add_argument("--min-attack-hours", type=float, default=1.0,
                        help="Minimum attack duration in hours to include")
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Trajectory Analysis on CERT Data")
    logger.info("=" * 60)
    
    # Load attack duration info
    data_dir = Path(args.data_dir)
    attack_info = get_attack_durations(data_dir)
    
    if len(attack_info) > 0:
        logger.info(f"\nAttack duration summary:")
        logger.info(f"  Total attacks: {len(attack_info)}")
        logger.info(f"  Under 1 hour: {(attack_info['duration_hours'] < 1).sum()}")
        logger.info(f"  1-24 hours: {((attack_info['duration_hours'] >= 1) & (attack_info['duration_hours'] < 24)).sum()}")
        logger.info(f"  1-7 days: {((attack_info['duration_hours'] >= 24) & (attack_info['duration_hours'] < 168)).sum()}")
        logger.info(f"  Over 7 days: {(attack_info['duration_hours'] >= 168).sum()}")
        
        # Filter attacks appropriate for our window
        suitable_attacks = attack_info[attack_info['duration_hours'] >= args.min_attack_hours]
        logger.info(f"\n  Attacks >= {args.min_attack_hours}hr (suitable for trajectory analysis): {len(suitable_attacks)}")
        suitable_users = set(suitable_attacks['user'].values)
    else:
        suitable_users = set()
    
    # Load processed data
    logger.info(f"\nLoading processed data from {args.load_processed}")
    data = np.load(args.load_processed)
    train_data = data['train_data']
    test_data = data['test_data']
    test_labels = data['test_labels']
    
    logger.info(f"  Train: {train_data.shape}, Test: {test_data.shape}")
    logger.info(f"  Malicious test samples: {test_labels.sum()}")
    
    # Load model
    logger.info(f"\nLoading model from {args.load_model}")
    checkpoint = torch.load(args.load_model, map_location='cpu', weights_only=False)
    
    model = UEBACNNAutoencoder(
        time_steps=checkpoint['time_steps'],
        n_features=checkpoint['n_features'],
        latent_dim=checkpoint['latent_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    mu = checkpoint['mu']
    sigma = checkpoint['sigma']
    
    logger.info(f"  Latent dim: {checkpoint['latent_dim']}")
    
    # Load manifold
    logger.info(f"\nLoading manifold from {args.load_manifold}")
    manifold_data = np.load(args.load_manifold)
    train_latents = manifold_data['train_latents']
    k_neighbors = int(manifold_data['k_neighbors'])
    
    manifold_config = UEBAManifoldConfig(k_neighbors=k_neighbors)
    manifold = UEBALatentManifold(train_latents, manifold_config)
    logger.info(f"  Manifold: {train_latents.shape[0]} points, k={k_neighbors}")
    
    # Create datasets
    train_sequences = [train_data[i] for i in range(len(train_data))]
    test_sequences = [test_data[i] for i in range(len(test_data))]
    
    train_dataset = SeqDataset(train_sequences, mu, sigma)
    test_dataset = SeqDataset(test_sequences, mu, sigma)
    
    # Extract test latents
    logger.info("\nExtracting test latent representations...")
    test_latents = []
    with torch.no_grad():
        for batch in DataLoader(test_dataset, batch_size=64):
            batch = batch.unsqueeze(1)
            latent = model.encode(batch)
            test_latents.append(latent.numpy())
    
    test_latents = np.vstack(test_latents)
    logger.info(f"  Test latents: {test_latents.shape}")
    
    # Initialize trajectory analyzer
    logger.info("\nInitializing trajectory analyzer...")
    traj_config = TrajectoryConfig(
        k_neighbors=16,
        min_trajectory_length=args.window_size
    )
    analyzer = TrajectoryAnalyzer(manifold, traj_config)
    
    # Fit reference statistics from training trajectories
    logger.info("Fitting reference statistics from training data...")
    train_trajectories = []
    stride = args.window_size // 2  # 50% overlap
    for i in range(0, len(train_latents) - args.window_size + 1, stride):
        traj = train_latents[i:i + args.window_size]
        train_trajectories.append(traj)
    
    analyzer.fit_reference_statistics(train_trajectories)
    logger.info(f"  Reference computed from {len(train_trajectories)} trajectories")
    
    # Analyze test trajectories
    logger.info("\n" + "=" * 60)
    logger.info("Trajectory Analysis Results")
    logger.info("=" * 60)
    
    # Separate normal and malicious latents
    normal_mask = test_labels == 0
    malicious_mask = test_labels == 1
    
    normal_latents = test_latents[normal_mask]
    malicious_latents = test_latents[malicious_mask]
    
    logger.info(f"\nTest set: {len(normal_latents)} normal, {len(malicious_latents)} malicious samples")
    
    # Create trajectories (sliding window)
    def create_trajectories(latents, window_size, stride):
        trajectories = []
        for i in range(0, len(latents) - window_size + 1, stride):
            trajectories.append(latents[i:i + window_size])
        return trajectories
    
    normal_trajectories = create_trajectories(normal_latents, args.window_size, stride)
    malicious_trajectories = create_trajectories(malicious_latents, args.window_size, stride)
    
    logger.info(f"Trajectories (window={args.window_size}, stride={stride}):")
    logger.info(f"  Normal: {len(normal_trajectories)}")
    logger.info(f"  Malicious: {len(malicious_trajectories)}")
    
    # Score trajectories
    logger.info("\nScoring trajectories...")
    
    normal_scores = []
    normal_features_list = []
    for traj in normal_trajectories:
        score, features = analyzer.score_trajectory(traj, return_features=True)
        normal_scores.append(score)
        if features['valid']:
            normal_features_list.append(features)
    
    malicious_scores = []
    malicious_features_list = []
    for traj in malicious_trajectories:
        score, features = analyzer.score_trajectory(traj, return_features=True)
        malicious_scores.append(score)
        if features['valid']:
            malicious_features_list.append(features)
    
    normal_scores = np.array(normal_scores)
    malicious_scores = np.array(malicious_scores)
    
    # Report results
    logger.info("\n" + "-" * 60)
    logger.info("Trajectory Anomaly Scores")
    logger.info("-" * 60)
    
    logger.info(f"\nNormal trajectories:")
    logger.info(f"  Mean: {np.mean(normal_scores):.4f}")
    logger.info(f"  Std:  {np.std(normal_scores):.4f}")
    logger.info(f"  Max:  {np.max(normal_scores):.4f}")
    
    logger.info(f"\nMalicious trajectories:")
    logger.info(f"  Mean: {np.mean(malicious_scores):.4f}")
    logger.info(f"  Std:  {np.std(malicious_scores):.4f}")
    logger.info(f"  Max:  {np.max(malicious_scores):.4f}")
    
    separation = (np.mean(malicious_scores) - np.mean(normal_scores)) / \
                 (np.std(normal_scores) + np.std(malicious_scores) + 1e-8)
    logger.info(f"\nSeparation score: {separation:.4f}")
    
    # Compute ROC-AUC and PR-AUC
    all_traj_scores = np.concatenate([normal_scores, malicious_scores])
    all_traj_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(malicious_scores))])
    
    traj_roc_auc = roc_auc_score(all_traj_labels, all_traj_scores)
    traj_pr_auc = average_precision_score(all_traj_labels, all_traj_scores)
    
    logger.info(f"\nTrajectory-based Classification Metrics:")
    logger.info(f"  ROC-AUC: {traj_roc_auc:.4f}")
    logger.info(f"  PR-AUC:  {traj_pr_auc:.4f}")
    
    # Feature breakdown
    logger.info("\n" + "-" * 60)
    logger.info("Feature Breakdown (mean values)")
    logger.info("-" * 60)
    
    feature_names = ['mean_velocity', 'mean_acceleration', 'max_manifold_distance', 
                     'manifold_drift_rate', 'tortuosity']
    
    logger.info(f"\n{'Feature':<25} {'Normal':<12} {'Malicious':<12} {'Ratio':<10}")
    logger.info("-" * 60)
    
    for fname in feature_names:
        normal_vals = [f[fname] for f in normal_features_list if fname in f]
        mal_vals = [f[fname] for f in malicious_features_list if fname in f]
        
        if normal_vals and mal_vals:
            n_mean = np.mean(normal_vals)
            m_mean = np.mean(mal_vals)
            ratio = m_mean / (n_mean + 1e-8)
            logger.info(f"{fname:<25} {n_mean:<12.4f} {m_mean:<12.4f} {ratio:<10.2f}x")
    
    # Compare to point-based
    logger.info("\n" + "-" * 60)
    logger.info("Comparison: Point-based vs Trajectory-based")
    logger.info("-" * 60)
    
    normal_point_dists = [manifold.normal_deviation(z) for z in normal_latents]
    malicious_point_dists = [manifold.normal_deviation(z) for z in malicious_latents]
    
    point_sep = (np.mean(malicious_point_dists) - np.mean(normal_point_dists)) / \
                (np.std(normal_point_dists) + np.std(malicious_point_dists) + 1e-8)
    
    # Point-based AUC metrics
    all_point_scores = np.concatenate([normal_point_dists, malicious_point_dists])
    all_point_labels = np.concatenate([np.zeros(len(normal_point_dists)), np.ones(len(malicious_point_dists))])
    
    point_roc_auc = roc_auc_score(all_point_labels, all_point_scores)
    point_pr_auc = average_precision_score(all_point_labels, all_point_scores)
    
    logger.info(f"\nPoint-based Classification Metrics:")
    logger.info(f"  ROC-AUC: {point_roc_auc:.4f}")
    logger.info(f"  PR-AUC:  {point_pr_auc:.4f}")
    
    logger.info(f"\nSeparation Scores:")
    logger.info(f"  Point-based:      {point_sep:.4f}")
    logger.info(f"  Trajectory-based: {separation:.4f}")
    
    logger.info(f"\n" + "-" * 60)
    logger.info("Summary: Point-based vs Trajectory-based")
    logger.info("-" * 60)
    logger.info(f"\n{'Metric':<20} {'Point-based':<15} {'Trajectory':<15} {'Improvement':<15}")
    logger.info("-" * 60)
    
    roc_imp = (traj_roc_auc - point_roc_auc) / point_roc_auc * 100
    pr_imp = (traj_pr_auc - point_pr_auc) / point_pr_auc * 100
    sep_imp = (separation - point_sep) / point_sep * 100
    
    logger.info(f"{'ROC-AUC':<20} {point_roc_auc:<15.4f} {traj_roc_auc:<15.4f} {roc_imp:+.1f}%")
    logger.info(f"{'PR-AUC':<20} {point_pr_auc:<15.4f} {traj_pr_auc:<15.4f} {pr_imp:+.1f}%")
    logger.info(f"{'Separation':<20} {point_sep:<15.4f} {separation:<15.4f} {sep_imp:+.1f}%")
    
    logger.info("\n" + "=" * 60)
    logger.info("Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
