"""
Plot PR curves for trajectory detection (Scenario 1 and 3).
Creates two separate figures for IEEE two-column format.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from pathlib import Path
import json

def topk_mean(scores, k):
    """Mean of top-k largest scores."""
    if len(scores) < k:
        return scores.mean()
    return np.partition(scores, -k)[-k:].mean()

def aggregate_window_scores_top2(trajectory_window_indices, window_scores):
    """Aggregate per-window scores using top-2 mean."""
    scores = np.zeros(len(trajectory_window_indices))
    for i, indices in enumerate(trajectory_window_indices):
        traj_scores = window_scores[indices]
        scores[i] = topk_mean(traj_scores, k=2)
    return scores

def plot_pr_curve_scenario(scores, labels, scenarios, scenario_id, scenario_name, output_path):
    """Plot PR curve for a specific scenario."""
    # Filter to this scenario vs all normals
    scenario_mask = (scenarios == scenario_id) | (labels == 0)
    scenario_scores = scores[scenario_mask]
    scenario_labels = labels[scenario_mask]
    
    # Compute PR curve
    precision, recall, _ = precision_recall_curve(scenario_labels, scenario_scores)
    pr_auc = average_precision_score(scenario_labels, scenario_scores)
    
    # Baseline (random classifier)
    baseline = scenario_labels.mean()
    
    # Create figure
    plt.figure(figsize=(6, 5))
    
    # Plot random baseline (faint)
    plt.plot([0, 1], [baseline, baseline], 'k--', linewidth=1, alpha=0.3, 
             label=f'Random (PR-AUC={baseline:.3f})')
    
    # Plot PR curve
    plt.plot(recall, precision, 'b-', linewidth=2, 
             label=f'Top-2 Mask Aggregation (PR-AUC={pr_auc:.3f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'{scenario_name}\nTrajectory-Level Detection (T=6 days)', fontsize=13, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    print(f"  {scenario_name}: PR-AUC={pr_auc:.4f}, Baseline={baseline:.4f}, N_pos={scenario_labels.sum()}")
    plt.close()

def main():
    exp_dir = Path('runs/exp015_unified_traj')
    
    print("="*80)
    print("Generating Trajectory PR Curves by Scenario")
    print("="*80)
    print()
    
    # Load window-level scores
    print("Loading data...")
    window_data = np.load(exp_dir / 'window_level_scores.npz', allow_pickle=True)
    mask_bce = window_data['mask_bce']
    
    # Load trajectory data
    traj_data = np.load(exp_dir / 'trajectory_level_scores.npz')
    label_any = traj_data['label_any']
    
    # Load config
    with open(exp_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    T = config['traj_window_size']
    stride = config['traj_stride']
    
    # Load test metadata
    from mv_ueba.etl.cert_fixed_window import CERTFixedWindowLoader
    
    data_dir = config['data_dir']
    loader = CERTFixedWindowLoader(data_dir=data_dir)
    _, test_data, test_labels, test_metadata, _ = loader.load_fixed_windows(
        bucket_hours=config['bucket_hours'],
        window_hours=config['window_hours'],
        buffer_days=config['buffer_days']
    )
    
    # Rebuild trajectories
    print("Building trajectories...")
    trajectory_window_indices = []
    trajectory_labels = []
    trajectory_scenarios = []
    
    for user_id, g in test_metadata.groupby("user_id", sort=False):
        g_sorted = g.sort_values("window_start")
        idx = g_sorted.index.to_numpy()
        
        y = test_labels[idx]
        scen = test_metadata.loc[idx, "scenario"].to_numpy()
        
        if len(idx) < T:
            continue
        
        for s in range(0, len(idx) - T + 1, stride):
            window_indices = idx[s:s + T]
            y_seq = y[s:s + T]
            scen_seq = scen[s:s + T]
            
            mal_frac_val = float(y_seq.mean())
            label_any_val = int(mal_frac_val > 0.0)
            
            if label_any_val:
                scenario_val = int(scen_seq[y_seq == 1][0])
            else:
                scenario_val = 0
            
            trajectory_window_indices.append(window_indices)
            trajectory_labels.append(label_any_val)
            trajectory_scenarios.append(scenario_val)
    
    trajectory_labels = np.array(trajectory_labels)
    trajectory_scenarios = np.array(trajectory_scenarios)
    
    print(f"  Built {len(trajectory_window_indices)} trajectories")
    print(f"  Scenario 1: {(trajectory_scenarios==1).sum()} trajectories")
    print(f"  Scenario 3: {(trajectory_scenarios==3).sum()} trajectories")
    print()
    
    # Compute top-2 mask aggregation
    print("Computing top-2 mask aggregation scores...")
    top2_scores = aggregate_window_scores_top2(trajectory_window_indices, mask_bce)
    print()
    
    # Plot Scenario 1
    print("Generating Figure A (Scenario 1)...")
    plot_pr_curve_scenario(
        top2_scores, 
        trajectory_labels, 
        trajectory_scenarios,
        scenario_id=1,
        scenario_name="Scenario 1: Insider Threat (CERT r4.2)",
        output_path=exp_dir / 'pr_curve_scenario1_top2.png'
    )
    print()
    
    # Plot Scenario 3
    print("Generating Figure B (Scenario 3)...")
    plot_pr_curve_scenario(
        top2_scores, 
        trajectory_labels, 
        trajectory_scenarios,
        scenario_id=3,
        scenario_name="Scenario 3: Insider Threat (CERT r4.2)",
        output_path=exp_dir / 'pr_curve_scenario3_top2.png'
    )
    print()
    
    print("="*80)
    print("Complete!")
    print("="*80)

if __name__ == '__main__':
    main()
