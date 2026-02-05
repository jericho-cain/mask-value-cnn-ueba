"""
Trajectory Analysis on CERT Dataset (User-Level Evaluation)

**STATUS: DEPRECATED - Superseded by segment_based_pipeline.py (exp004)**

This script implements trajectory-based detection with user-level evaluation
used in exp002/exp003. Preserved for reproducing baseline experiments.

**Why deprecated:**
- User-level evaluation (50 normal vs 70 malicious users)
- Low temporal precision (9.9% in exp003)
- Doesn't reflect production deployment

**For new experiments, use:** `examples/segment_based_pipeline.py`
**See:** EXPERIMENTS.md for experiment evolution

---

This script tests geodesic deviation (trajectory-based anomaly detection)
on CERT insider threat data with user-level train/test split.

Usage (for reproducing exp002/exp003 only):
    python examples/trajectory_analysis.py --load-experiment runs/exp002_baseline

For full details on experiment history and current approach, see EXPERIMENTS.md
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix

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


def load_attack_windows(data_dir: Path) -> dict:
    """Load attack time windows from CERT answers file.
    
    Returns dict mapping user_id -> {'start': datetime, 'end': datetime, 'scenario': int}
    """
    attacks_df = get_attack_durations(data_dir)
    if attacks_df.empty:
        return {}
    
    # Build mapping
    attack_windows = {}
    for _, row in attacks_df.iterrows():
        user = row["user"]
        attack_windows[user] = {
            'start': row["start_dt"].to_pydatetime(),
            'end': row["end_dt"].to_pydatetime(),
            'scenario': int(row["scenario"])
        }
    
    logger.info(f"Loaded attack windows for {len(attack_windows)} malicious users")
    return attack_windows


def parse_args():
    parser = argparse.ArgumentParser(description="Trajectory analysis on CERT data")
    parser.add_argument("--load-processed", type=str, default=None,
                        help="Path to processed data .npz file")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to trained model .pt file")
    parser.add_argument("--load-manifold", type=str, default=None,
                        help="Path to manifold .npz file")
    parser.add_argument("--load-experiment", type=str, default=None,
                        help="Load from experiment directory (auto-sets load paths)")
    parser.add_argument("--data-dir", type=str, default="data/cert/r4.2",
                        help="Path to CERT data (for attack duration info)")
    parser.add_argument("--window-size", type=int, default=6,
                        help="Number of consecutive latent points per trajectory")
    parser.add_argument("--stride", type=int, default=None,
                        help="Step size between trajectory windows. Default: window_size // 2 (50%% overlap)")
    parser.add_argument("--min-attack-hours", type=float, default=1.0,
                        help="Minimum attack duration in hours to include")
    return parser.parse_args()


def setup_experiment(args) -> Path | None:
    """Setup experiment paths from --load-experiment."""
    if not args.load_experiment:
        # Check that required paths are provided
        if not all([args.load_processed, args.load_model, args.load_manifold]):
            raise ValueError("Either --load-experiment or all of --load-processed, --load-model, --load-manifold required")
        return None
    
    exp_dir = Path(args.load_experiment)
    if not exp_dir.exists():
        raise ValueError(f"Experiment directory not found: {exp_dir}")
    
    # Load config if exists
    config_path = exp_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        logger.info(f"Loaded experiment config from {config_path}")
    
    # Set load paths from experiment directory
    if not args.load_processed:
        processed_path = exp_dir / "processed.npz"
        if processed_path.exists():
            args.load_processed = str(processed_path)
        else:
            raise ValueError(f"processed.npz not found in {exp_dir}")
    
    if not args.load_model:
        model_path = exp_dir / "model.pt"
        if model_path.exists():
            args.load_model = str(model_path)
        else:
            logger.warning(f"model.pt not found in {exp_dir}, will need to be specified explicitly")
    
    if not args.load_manifold:
        manifold_path = exp_dir / "manifold.npz"
        if manifold_path.exists():
            args.load_manifold = str(manifold_path)
        else:
            logger.warning(f"manifold.npz not found in {exp_dir}, will need to be specified explicitly")
    
    logger.info(f"Loading from experiment: {exp_dir}")
    return exp_dir


def main():
    args = parse_args()
    
    # Setup experiment if specified
    exp_dir = setup_experiment(args)
    
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
    data = np.load(args.load_processed, allow_pickle=True)
    train_data = data['train_data']
    test_data = data['test_data']
    test_labels = data['test_labels']
    
    # Load user IDs, timestamps, and scenarios if available
    test_user_ids = data['test_user_ids'] if 'test_user_ids' in data else None
    test_timestamps = data['test_timestamps'] if 'test_timestamps' in data else None
    test_scenarios = data['test_scenarios'] if 'test_scenarios' in data else None
    
    logger.info(f"  Train: {train_data.shape}, Test: {test_data.shape}")
    logger.info(f"  Malicious test samples: {test_labels.sum()}")
    if test_user_ids is not None:
        n_malicious_users = len(np.unique(test_user_ids[test_labels == 1]))
        n_normal_users = len(np.unique(test_user_ids[test_labels == 0]))
        logger.info(f"  Unique users in test: {n_normal_users} normal, {n_malicious_users} malicious")
    if test_scenarios is not None:
        for s in [1, 2, 3]:
            count = (test_scenarios == s).sum()
            if count > 0:
                logger.info(f"  Scenario {s}: {count} sequences")
    
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
    stride = args.stride if args.stride is not None else args.window_size // 2
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
    
    # Get user IDs and scenarios for normal/malicious if available
    normal_user_ids = test_user_ids[normal_mask] if test_user_ids is not None else None
    malicious_user_ids = test_user_ids[malicious_mask] if test_user_ids is not None else None
    malicious_scenarios_arr = test_scenarios[malicious_mask] if test_scenarios is not None else None
    
    logger.info(f"\nTest set: {len(normal_latents)} normal, {len(malicious_latents)} malicious samples")
    
    # Create trajectories (sliding window) with user ID, scenario, and timestamp tracking
    def create_trajectories_with_metadata(latents, user_ids, scenarios, timestamps, window_size, stride):
        """Create trajectories and track which user/scenario/timestamp each trajectory belongs to."""
        trajectories = []
        traj_user_ids = []
        traj_scenarios = []
        traj_timestamps = []
        for i in range(0, len(latents) - window_size + 1, stride):
            trajectories.append(latents[i:i + window_size])
            # Assign trajectory to user/scenario/timestamp of the first sample in window
            if user_ids is not None:
                traj_user_ids.append(user_ids[i])
            if scenarios is not None:
                traj_scenarios.append(scenarios[i])
            if timestamps is not None:
                traj_timestamps.append(timestamps[i])
        return (
            trajectories,
            traj_user_ids if user_ids is not None else None,
            traj_scenarios if scenarios is not None else None,
            traj_timestamps if timestamps is not None else None
        )
    
    normal_traj_timestamps = test_timestamps[normal_mask] if test_timestamps is not None else None
    malicious_traj_timestamps_arr = test_timestamps[malicious_mask] if test_timestamps is not None else None
    
    normal_trajectories, normal_traj_users, _, normal_traj_ts = create_trajectories_with_metadata(
        normal_latents, normal_user_ids, None, normal_traj_timestamps, args.window_size, stride
    )
    malicious_trajectories, malicious_traj_users, malicious_traj_scenarios, malicious_traj_ts = create_trajectories_with_metadata(
        malicious_latents, malicious_user_ids, malicious_scenarios_arr, malicious_traj_timestamps_arr, args.window_size, stride
    )
    
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
    
    # Confusion matrix at optimal F1 threshold
    logger.info("\n" + "-" * 60)
    logger.info("Confusion Matrix at Optimal F1 Threshold")
    logger.info("-" * 60)
    
    def compute_optimal_f1_confusion(y_true, scores, method_name):
        """Find optimal F1 threshold and return confusion matrix."""
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        # F1 = 2 * (precision * recall) / (precision + recall)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]
        
        # Compute predictions at optimal threshold
        predictions = (scores >= best_threshold).astype(int)
        cm = confusion_matrix(y_true, predictions)
        
        logger.info(f"\n{method_name}:")
        logger.info(f"  Optimal threshold: {best_threshold:.4f}")
        logger.info(f"  F1: {best_f1:.4f}, Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")
        logger.info(f"  Confusion Matrix:")
        logger.info(f"                    Predicted")
        logger.info(f"                 Normal  Malicious")
        logger.info(f"    Actual Normal   {cm[0,0]:5d}    {cm[0,1]:5d}")
        logger.info(f"    Actual Malicious{cm[1,0]:5d}    {cm[1,1]:5d}")
        
        return cm, best_f1, best_threshold
    
    point_cm, point_f1, point_thresh = compute_optimal_f1_confusion(all_point_labels, all_point_scores, "Point-based")
    traj_cm, traj_f1, traj_thresh = compute_optimal_f1_confusion(all_traj_labels, all_traj_scores, "Trajectory-based")
    
    # User-level metrics (if user IDs available)
    user_level_results = None
    if malicious_traj_users is not None and normal_traj_users is not None:
        logger.info("\n" + "-" * 60)
        logger.info("User-Level Detection Metrics")
        logger.info("-" * 60)
        
        # For each user, check if ANY of their trajectories exceed threshold
        def compute_user_level_detection(traj_scores, traj_users, threshold):
            """Compute which users have at least one trajectory above threshold."""
            users_detected = set()
            all_users = set(traj_users)
            
            for score, user in zip(traj_scores, traj_users):
                if score >= threshold:
                    users_detected.add(user)
            
            return users_detected, all_users
        
        # At trajectory optimal threshold
        malicious_users_detected, all_malicious_users = compute_user_level_detection(
            malicious_scores, malicious_traj_users, traj_thresh
        )
        normal_users_flagged, all_normal_users = compute_user_level_detection(
            normal_scores, normal_traj_users, traj_thresh
        )
        
        n_malicious_detected = len(malicious_users_detected)
        n_malicious_total = len(all_malicious_users)
        n_normal_flagged = len(normal_users_flagged)
        n_normal_total = len(all_normal_users)
        
        user_recall = n_malicious_detected / n_malicious_total if n_malicious_total > 0 else 0
        user_fpr = n_normal_flagged / n_normal_total if n_normal_total > 0 else 0
        
        logger.info(f"\nAt trajectory optimal threshold ({traj_thresh:.4f}):")
        logger.info(f"  Malicious users detected: {n_malicious_detected}/{n_malicious_total} ({user_recall:.1%})")
        logger.info(f"  Normal users falsely flagged: {n_normal_flagged}/{n_normal_total} ({user_fpr:.1%})")
        
        # List which malicious users were missed (if any)
        missed_users = all_malicious_users - malicious_users_detected
        if missed_users:
            logger.info(f"  Missed malicious users: {sorted(missed_users)}")
        
        user_level_results = {
            'threshold': float(traj_thresh),
            'n_malicious_users_total': n_malicious_total,
            'n_malicious_users_detected': n_malicious_detected,
            'user_recall': float(user_recall),
            'n_normal_users_total': n_normal_total,
            'n_normal_users_flagged': n_normal_flagged,
            'user_fpr': float(user_fpr),
            'missed_users': list(missed_users) if missed_users else [],
        }
    
    # Scenario-level metrics (if scenarios available)
    scenario_level_results = None
    if malicious_traj_scenarios is not None:
        logger.info("\n" + "-" * 60)
        logger.info("Scenario-Level Detection Metrics")
        logger.info("-" * 60)
        
        scenario_results = {}
        for scenario in sorted(set(malicious_traj_scenarios)):
            # Get trajectories for this scenario
            scenario_mask = [s == scenario for s in malicious_traj_scenarios]
            scenario_scores = [malicious_scores[i] for i, m in enumerate(scenario_mask) if m]
            scenario_users = [malicious_traj_users[i] for i, m in enumerate(scenario_mask) if m] if malicious_traj_users else None
            
            n_traj = len(scenario_scores)
            n_detected = sum(1 for s in scenario_scores if s >= traj_thresh)
            traj_recall = n_detected / n_traj if n_traj > 0 else 0
            
            # User-level for this scenario
            if scenario_users:
                users_in_scenario = set(scenario_users)
                users_detected = set()
                for score, user in zip(scenario_scores, scenario_users):
                    if score >= traj_thresh:
                        users_detected.add(user)
                n_users = len(users_in_scenario)
                n_users_detected = len(users_detected)
                user_recall_scenario = n_users_detected / n_users if n_users > 0 else 0
                missed = users_in_scenario - users_detected
            else:
                n_users = n_users_detected = 0
                user_recall_scenario = 0
                missed = set()
            
            logger.info(f"\nScenario {scenario}:")
            logger.info(f"  Trajectories: {n_detected}/{n_traj} detected ({traj_recall:.1%})")
            if scenario_users:
                logger.info(f"  Users: {n_users_detected}/{n_users} detected ({user_recall_scenario:.1%})")
                if missed:
                    logger.info(f"  Missed users: {sorted(missed)}")
            
            scenario_results[f"scenario_{scenario}"] = {
                'n_trajectories': n_traj,
                'n_trajectories_detected': n_detected,
                'trajectory_recall': float(traj_recall),
                'n_users': n_users,
                'n_users_detected': n_users_detected,
                'user_recall': float(user_recall_scenario),
                'missed_users': list(missed) if missed else [],
            }
        
        scenario_level_results = scenario_results
    
    # Temporal evaluation: match detections against attack windows
    temporal_results = None
    if test_timestamps is not None and test_user_ids is not None:
        logger.info("\n" + "-" * 60)
        logger.info("Temporal Precision Evaluation")
        logger.info("-" * 60)
        
        # Load attack windows
        attack_windows = load_attack_windows(Path(args.data_dir))
        
        if attack_windows and malicious_traj_ts is not None:
            # Calculate trajectory span (window_size sequences, each 24 hours)
            sequence_duration = timedelta(hours=24)  # Assuming 24-hour sequences
            trajectory_window_span = args.window_size * sequence_duration
            
            # Classify trajectories as during-attack or not
            attack_period_mask = []
            non_attack_period_mask = []
            
            for i, (user, score, ts) in enumerate(zip(malicious_traj_users, malicious_scores, malicious_traj_ts)):
                if ts is None or user not in attack_windows:
                    continue
                
                attack_window = attack_windows[user]
                # Check if trajectory overlaps with attack window
                # Trajectory spans from ts to ts + trajectory_window_span
                traj_start = pd.Timestamp(ts).to_pydatetime()
                traj_end = traj_start + trajectory_window_span
                
                # Overlap if: traj_start < attack_end AND traj_end > attack_start
                is_during_attack = (traj_start < attack_window['end'] and 
                                   traj_end > attack_window['start'])
                
                if is_during_attack:
                    attack_period_mask.append(i)
                else:
                    non_attack_period_mask.append(i)
            
            # Compute metrics for attack-period vs non-attack-period
            if attack_period_mask:
                attack_scores = [malicious_scores[i] for i in attack_period_mask]
                n_attack_detected = sum(1 for s in attack_scores if s >= traj_thresh)
                attack_recall = n_attack_detected / len(attack_scores) if attack_scores else 0
                
                logger.info(f"\nTrajectories during attack periods:")
                logger.info(f"  Total: {len(attack_scores)}")
                logger.info(f"  Detected: {n_attack_detected} ({attack_recall:.1%})")
            
            if non_attack_period_mask:
                non_attack_scores = [malicious_scores[i] for i in non_attack_period_mask]
                n_non_attack_detected = sum(1 for s in non_attack_scores if s >= traj_thresh)
                non_attack_rate = n_non_attack_detected / len(non_attack_scores) if non_attack_scores else 0
                
                logger.info(f"\nTrajectories outside attack periods (normal behavior of malicious users):")
                logger.info(f"  Total: {len(non_attack_scores)}")
                logger.info(f"  Flagged: {n_non_attack_detected} ({non_attack_rate:.1%})")
            
            # Temporal precision: of all flagged trajectories, what % were during attacks?
            all_flagged_indices = [i for i, s in enumerate(malicious_scores) if s >= traj_thresh]
            flagged_during_attack = len(set(all_flagged_indices) & set(attack_period_mask))
            temporal_precision = flagged_during_attack / len(all_flagged_indices) if all_flagged_indices else 0
            
            logger.info(f"\nTemporal Precision:")
            logger.info(f"  Of {len(all_flagged_indices)} flagged malicious trajectories:")
            logger.info(f"    {flagged_during_attack} were during attack periods ({temporal_precision:.1%})")
            logger.info(f"    {len(all_flagged_indices) - flagged_during_attack} were during normal periods")
            
            temporal_results = {
                'n_attack_period_trajectories': len(attack_period_mask),
                'n_attack_period_detected': n_attack_detected if attack_period_mask else 0,
                'attack_period_recall': float(attack_recall) if attack_period_mask else 0.0,
                'n_non_attack_period_trajectories': len(non_attack_period_mask),
                'n_non_attack_period_flagged': n_non_attack_detected if non_attack_period_mask else 0,
                'non_attack_period_fpr': float(non_attack_rate) if non_attack_period_mask else 0.0,
                'temporal_precision': float(temporal_precision),
            }
    
    # Save trajectory results to experiment directory
    if exp_dir:
        traj_results = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'window_size': args.window_size,
                'stride': stride,
                'min_attack_hours': args.min_attack_hours,
            },
            'data_stats': {
                'n_normal_samples': int(normal_mask.sum()),
                'n_malicious_samples': int(malicious_mask.sum()),
                'n_normal_trajectories': len(normal_scores),
                'n_malicious_trajectories': len(malicious_scores),
            },
            'point_based': {
                'roc_auc': float(point_roc_auc),
                'pr_auc': float(point_pr_auc),
                'separation': float(point_sep),
                'optimal_f1': float(point_f1),
                'optimal_threshold': float(point_thresh),
                'confusion_matrix': point_cm.tolist(),
            },
            'trajectory_based': {
                'roc_auc': float(traj_roc_auc),
                'pr_auc': float(traj_pr_auc),
                'separation': float(separation),
                'optimal_f1': float(traj_f1),
                'optimal_threshold': float(traj_thresh),
                'confusion_matrix': traj_cm.tolist(),
            },
            'improvement': {
                'roc_auc_pct': float(roc_imp),
                'pr_auc_pct': float(pr_imp),
                'separation_pct': float(sep_imp),
            }
        }
        
        # Add user-level results if available
        if user_level_results is not None:
            traj_results['user_level'] = user_level_results
        
        # Add scenario-level results if available
        if scenario_level_results is not None:
            traj_results['scenario_level'] = scenario_level_results
        
        # Add temporal results if available
        if temporal_results is not None:
            traj_results['temporal_evaluation'] = temporal_results
        
        traj_results_path = exp_dir / "trajectory_results.json"
        with open(traj_results_path, 'w') as f:
            json.dump(traj_results, f, indent=2)
        logger.info(f"\nSaved trajectory results to {traj_results_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
