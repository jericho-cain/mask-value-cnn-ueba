"""
Phase 1: Energy Baseline Aggregators

Computes trajectory-level scores by aggregating per-window anomaly scores.
Tests hypothesis: sparse anomalies (2-3 bad days) are better captured by
top-k aggregation than by geodesic deviation.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score

def topk_mean(scores, k):
    """Mean of top-k largest scores."""
    if len(scores) < k:
        return scores.mean()
    return np.partition(scores, -k)[-k:].mean()

def aggregate_window_scores(trajectory_window_indices, window_scores):
    """
    Aggregate per-window scores to trajectory-level.
    
    Args:
        trajectory_window_indices: list of arrays, each containing global window indices
        window_scores: (N_windows,) array of per-window scores
        
    Returns:
        dict of trajectory-level scores for different aggregators
    """
    n_trajs = len(trajectory_window_indices)
    
    scores = {
        'sum': np.zeros(n_trajs),
        'mean': np.zeros(n_trajs),
        'top2': np.zeros(n_trajs),
        'top3': np.zeros(n_trajs),
    }
    
    for i, indices in enumerate(trajectory_window_indices):
        traj_scores = window_scores[indices]
        scores['sum'][i] = traj_scores.sum()
        scores['mean'][i] = traj_scores.mean()
        scores['top2'][i] = topk_mean(traj_scores, k=2)
        scores['top3'][i] = topk_mean(traj_scores, k=3)
    
    return scores

def evaluate_scores(scores, labels, method_name, scenarios=None):
    """Compute ROC-AUC and PR-AUC for trajectory scores."""
    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    
    # Best F1
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    
    result = {
        'method': method_name,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'best_f1': best_f1,
    }
    
    # Add per-scenario breakdown if available
    if scenarios is not None:
        for scenario_id in [1, 3]:
            # Get indices for this scenario (positives) vs all normals
            scenario_mask = (scenarios == scenario_id) | (labels == 0)
            if scenario_mask.sum() > 0 and (labels[scenario_mask] == 1).sum() > 0:
                scenario_scores = scores[scenario_mask]
                scenario_labels = labels[scenario_mask]
                
                try:
                    scen_roc = roc_auc_score(scenario_labels, scenario_scores)
                    scen_pr = average_precision_score(scenario_labels, scenario_scores)
                    result[f'scenario_{scenario_id}_roc_auc'] = scen_roc
                    result[f'scenario_{scenario_id}_pr_auc'] = scen_pr
                except:
                    result[f'scenario_{scenario_id}_roc_auc'] = 0.0
                    result[f'scenario_{scenario_id}_pr_auc'] = 0.0
    
    return result

def main():
    exp_dir = Path('runs/exp015_unified_traj')
    
    print("="*80)
    print("Phase 1: Energy Baseline Aggregators")
    print("="*80)
    print()
    
    # Load window-level scores
    print("Loading window-level scores from exp015...")
    window_data = np.load(exp_dir / 'window_level_scores.npz', allow_pickle=True)
    
    # Available window-level scores
    mask_bce = window_data['mask_bce']
    ae_total = window_data['ae_total']
    beta = window_data['off_manifold_distances']
    
    print(f"  mask_bce: shape={mask_bce.shape}, range=[{mask_bce.min():.4f}, {mask_bce.max():.4f}]")
    print(f"  ae_total: shape={ae_total.shape}, range=[{ae_total.min():.4f}, {ae_total.max():.4f}]")
    print(f"  beta: shape={beta.shape}, range=[{beta.min():.4f}, {beta.max():.4f}]")
    print()
    
    # Load trajectory-level data
    print("Loading trajectory-level data...")
    traj_data = np.load(exp_dir / 'trajectory_level_scores.npz')
    
    label_any = traj_data['label_any']
    label_majority = traj_data['label_majority']
    mal_frac = traj_data['mal_frac']
    geodesic_scores = traj_data['trajectory_scores']
    
    n_trajs = len(label_any)
    print(f"  Trajectories: {n_trajs}")
    print(f"  Any-overlap positives: {label_any.sum()} ({label_any.mean()*100:.1f}%)")
    print(f"  Majority-overlap positives: {label_majority.sum()} ({label_majority.mean()*100:.1f}%)")
    print()
    
    # Build trajectory window indices from test metadata
    # We need to reconstruct which windows belong to which trajectory
    print("Reconstructing trajectory-to-window mapping...")
    
    # Load test metadata to rebuild trajectories
    with open(exp_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    T = config['traj_window_size']
    stride = config['traj_stride']
    
    # We need to load the original test metadata
    # For now, let's use a simpler approach: compute aggregations directly
    # by matching trajectory structure
    
    # Since we don't have direct window-to-traj mapping saved, let's compute it
    # The test set has 1609 windows, and we have 501 trajectories
    # We need to reconstruct the sliding window structure
    
    # Load the test metadata to get user_id structure
    from manifold_ueba.etl.cert_fixed_window import CERTFixedWindowLoader
    
    data_dir = config['data_dir']
    loader = CERTFixedWindowLoader(data_dir=data_dir)
    _, test_data, test_labels, test_metadata, _ = loader.load_fixed_windows(
        bucket_hours=config['bucket_hours'],
        window_hours=config['window_hours'],
        buffer_days=config['buffer_days']
    )
    
    print(f"  Test windows: {len(test_data)}")
    print(f"  Test metadata shape: {test_metadata.shape}")
    print()
    
    # Build trajectory-to-window mapping using same logic as in pipeline
    print("Building trajectories with per-user sliding windows...")
    
    trajectory_window_indices = []
    trajectory_labels_any = []
    trajectory_labels_maj = []
    trajectory_mal_fracs = []
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
            label_maj_val = int(mal_frac_val >= 0.5)
            
            # Scenario attribution: first malicious window's scenario, or 0 if all normal
            if label_any_val:
                scenario_val = int(scen_seq[y_seq == 1][0])
            else:
                scenario_val = 0
            
            trajectory_window_indices.append(window_indices)
            trajectory_labels_any.append(label_any_val)
            trajectory_labels_maj.append(label_maj_val)
            trajectory_mal_fracs.append(mal_frac_val)
            trajectory_scenarios.append(scenario_val)
    
    trajectory_labels_any = np.array(trajectory_labels_any)
    trajectory_labels_maj = np.array(trajectory_labels_maj)
    trajectory_mal_fracs = np.array(trajectory_mal_fracs)
    trajectory_scenarios = np.array(trajectory_scenarios)
    
    print(f"  Built {len(trajectory_window_indices)} trajectories")
    print(f"  Sanity check: {trajectory_labels_any.sum()} any-overlap positives (expected {label_any.sum()})")
    print(f"  Scenario breakdown: scenario_1={(trajectory_scenarios==1).sum()}, scenario_3={(trajectory_scenarios==3).sum()}")
    print()
    
    # Compute aggregated scores for each window-level score type
    print("Computing aggregated trajectory scores...")
    print()
    
    aggregators = ['sum', 'mean', 'top2', 'top3']
    window_score_types = {
        'mask_bce': mask_bce,
        'ae_total': ae_total,
        'beta': beta,
    }
    
    all_results = []
    
    for score_name, window_scores in window_score_types.items():
        print(f"--- {score_name.upper()} ---")
        
        # Compute aggregations
        agg_scores = aggregate_window_scores(trajectory_window_indices, window_scores)
        
        # Evaluate each aggregator on both label semantics
        for agg_name in aggregators:
            scores = agg_scores[agg_name]
            
            # Any-overlap
            res_any = evaluate_scores(scores, trajectory_labels_any, f"{score_name}_{agg_name}_any", 
                                     scenarios=trajectory_scenarios)
            all_results.append(res_any)
            
            # Majority-overlap
            res_maj = evaluate_scores(scores, trajectory_labels_maj, f"{score_name}_{agg_name}_maj",
                                     scenarios=trajectory_scenarios)
            all_results.append(res_maj)
            
            print(f"  {agg_name:6s}: Any PR-AUC={res_any['pr_auc']:.4f}, Maj PR-AUC={res_maj['pr_auc']:.4f}")
        
        print()
    
    # Add geodesic baseline for comparison
    print("--- GEODESIC (baseline) ---")
    geo_any = evaluate_scores(geodesic_scores, trajectory_labels_any, "geodesic_any",
                              scenarios=trajectory_scenarios)
    geo_maj = evaluate_scores(geodesic_scores, trajectory_labels_maj, "geodesic_maj",
                              scenarios=trajectory_scenarios)
    all_results.extend([geo_any, geo_maj])
    print(f"  baseline: Any PR-AUC={geo_any['pr_auc']:.4f}, Maj PR-AUC={geo_maj['pr_auc']:.4f}")
    print()
    
    # Create summary table
    print("="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print()
    
    # Convert to DataFrame for nice formatting
    df = pd.DataFrame(all_results)
    
    # Split into any and majority
    df_any = df[df['method'].str.contains('_any')].copy()
    df_maj = df[df['method'].str.contains('_maj')].copy()
    
    df_any['method'] = df_any['method'].str.replace('_any', '')
    df_maj['method'] = df_maj['method'].str.replace('_maj', '')
    
    # Merge
    df_summary = pd.merge(
        df_any[['method', 'pr_auc', 'roc_auc']], 
        df_maj[['method', 'pr_auc', 'roc_auc']], 
        on='method', 
        suffixes=('_any', '_maj')
    )
    
    # Sort by any-overlap PR-AUC
    df_summary = df_summary.sort_values('pr_auc_any', ascending=False)
    
    print("Method                    | Any PR-AUC | Any ROC-AUC | Maj PR-AUC | Maj ROC-AUC")
    print("-" * 80)
    for _, row in df_summary.iterrows():
        print(f"{row['method']:25s} | {row['pr_auc_any']:10.4f} | {row['roc_auc_any']:11.4f} | "
              f"{row['pr_auc_maj']:10.4f} | {row['roc_auc_maj']:11.4f}")
    
    print()
    
    # Per-scenario breakdown table
    print("="*80)
    print("PER-SCENARIO BREAKDOWN (Any-Overlap)")
    print("="*80)
    print()
    
    print("Method                    | Scenario 1 PR-AUC | Scenario 1 ROC-AUC | Scenario 3 PR-AUC | Scenario 3 ROC-AUC")
    print("-" * 110)
    
    # Get top methods for scenario breakdown
    top_methods = df_summary.head(10)['method'].values
    
    for method_name in top_methods:
        res = [r for r in all_results if r['method'] == f"{method_name}_any"][0]
        scen1_pr = res.get('scenario_1_pr_auc', 0.0)
        scen1_roc = res.get('scenario_1_roc_auc', 0.0)
        scen3_pr = res.get('scenario_3_pr_auc', 0.0)
        scen3_roc = res.get('scenario_3_roc_auc', 0.0)
        print(f"{method_name:25s} | {scen1_pr:17.4f} | {scen1_roc:18.4f} | {scen3_pr:17.4f} | {scen3_roc:18.4f}")
    
    print()
    
    # Save results
    output_file = exp_dir / 'phase1_aggregation_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print()
    
    # Key findings
    print("="*80)
    print("KEY FINDINGS")
    print("="*80)
    print()
    
    best_any = df_summary.iloc[0]
    baseline_any_prauc = df_summary[df_summary['method'] == 'geodesic']['pr_auc_any'].values[0]
    improvement = (best_any['pr_auc_any'] - baseline_any_prauc) / baseline_any_prauc * 100
    
    print(f"Best aggregator: {best_any['method']}")
    print(f"  Any-overlap PR-AUC: {best_any['pr_auc_any']:.4f} (vs geodesic: {baseline_any_prauc:.4f})")
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"  Majority-overlap PR-AUC: {best_any['pr_auc_maj']:.4f}")
    print()
    
    if 'top' in best_any['method']:
        print("Confirms hypothesis: Sparse anomalies (2-3 bad days) are better captured")
        print("by top-k aggregation than by trajectory-shape metrics.")
    else:
        print("Unexpected: Mean/sum outperforms top-k. May indicate distributed anomalies.")
    
    print()

if __name__ == '__main__':
    main()
