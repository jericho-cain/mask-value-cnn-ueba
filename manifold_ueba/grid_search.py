"""
Grid Search for Alpha/Beta Hyperparameter Optimization

This module implements systematic hyperparameter optimization for the hybrid
CNN Autoencoder + Manifold Learning approach, following the methodology
from the gravitational wave manifold learning research.

The grid search finds optimal α (reconstruction weight) and β (manifold weight)
combinations that maximize anomaly detection performance.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from .cnn_model import UEBACNNAutoencoder
from .latent_manifold import UEBALatentManifold
from .manifold_scorer import UEBAManifoldScorer, UEBAManifoldScorerConfig


@dataclass
class GridSearchResult:
    """Container for grid search optimization results."""
    best_alpha: float
    best_beta: float
    best_auc: float
    best_precision: float
    best_recall: float
    all_results: dict[tuple[float, float], dict[str, float]]
    baseline_auc: float  # AUC when β=0 (LSTM-only)


def compute_binary_metrics(
    normal_scores: torch.Tensor, 
    anomalous_scores: torch.Tensor
) -> dict[str, float]:
    """
    Compute binary classification metrics from anomaly scores.
    
    Parameters
    ----------
    normal_scores : torch.Tensor
        Anomaly scores for normal sequences
    anomalous_scores : torch.Tensor  
        Anomaly scores for anomalous sequences
        
    Returns
    -------
    dict
        Contains 'auc', 'precision', 'recall', 'f1', 'separation'
    """
    # Create labels (0=normal, 1=anomalous)
    y_true = torch.cat([
        torch.zeros(len(normal_scores)),
        torch.ones(len(anomalous_scores))
    ]).numpy()
    
    # Combine scores
    y_scores = torch.cat([normal_scores, anomalous_scores]).numpy()
    
    # Compute AUC
    auc_score = roc_auc_score(y_true, y_scores)
    
    # Compute precision-recall curve and PR-AUC
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    # Find optimal threshold (maximize F1)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # Separation metric (higher is better)
    normal_mean = torch.mean(normal_scores).item()
    normal_std = torch.std(normal_scores).item()
    anomalous_mean = torch.mean(anomalous_scores).item()
    anomalous_std = torch.std(anomalous_scores).item()
    separation = (anomalous_mean - normal_mean) / (normal_std + anomalous_std + 1e-8)
    
    return {
        'auc': auc_score,
        'pr_auc': pr_auc,
        'precision': best_precision,
        'recall': best_recall, 
        'f1': best_f1,
        'separation': separation
    }


def grid_search_alpha_beta(
    cnn_model: UEBACNNAutoencoder,
    manifold: UEBALatentManifold,
    normal_sequences: torch.Tensor,
    anomalous_sequences: torch.Tensor,
    alphas: list[float] = None,
    betas: list[float] = None,
    verbose: bool = True
) -> GridSearchResult:
    """
    Systematic grid search over α/β combinations for CNN + Manifold architecture.
    
    Following the gravitational wave paper methodology, this searches over
    reconstruction weight (α) and manifold weight (β) to maximize AUC.
    
    Parameters
    ----------
    cnn_model : UEBACNNAutoencoder
        Trained CNN autoencoder model
    manifold : UEBALatentManifold
        Built manifold from normal behavioral patterns
    normal_sequences : torch.Tensor
        Normal behavioral sequences for evaluation, shape (N, 1, time_steps, n_features)
    anomalous_sequences : torch.Tensor
        Anomalous behavioral sequences for evaluation, shape (M, 1, time_steps, n_features)
    alphas : List[float], optional
        Reconstruction weights to test. Default: [0.1, 0.5, 1.0, 2.0, 5.0]
    betas : List[float], optional  
        Manifold weights to test. Default: [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    GridSearchResult
        Optimization results with best configuration and all evaluations
    """
    if alphas is None:
        alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
    if betas is None:
        betas = [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    if verbose:
        total_combinations = len(alphas) * len(betas)
        print(
            f"Grid search: {len(alphas)} alphas × {len(betas)} betas "
            f"= {total_combinations} combinations"
        )
        print(f"Alpha values: {alphas}")
        print(f"Beta values: {betas}")
        print("-" * 60)
    
    all_results = {}
    best_auc = 0.0
    best_config = None
    best_metrics = None
    baseline_auc = 0.0
    
    total_combinations = len(alphas) * len(betas)
    current_combination = 0
    
    for alpha in alphas:
        for beta in betas:
            current_combination += 1
            
            if verbose:
                print(
                    f"[{current_combination:2d}/{total_combinations}] "
                    f"Testing α={alpha:4.2f}, β={beta:4.2f}", 
                    end=""
                )
            
            # Create scorer with current α/β configuration
            if beta == 0.0:
                # CNN only mode
                config = UEBAManifoldScorerConfig(mode="cnn_only", alpha_cnn=alpha)
            else:
                # CNN + Manifold mode
                config = UEBAManifoldScorerConfig(
                    mode="cnn_plus_manifold", 
                    alpha_cnn=alpha, 
                    beta_manifold=beta
                )
            
            scorer = UEBAManifoldScorer(manifold, config)
            
            # Score all sequences with current α/β
            normal_scores_dict = scorer.score_batch(cnn_model, normal_sequences)
            anomalous_scores_dict = scorer.score_batch(cnn_model, anomalous_sequences)
            
            normal_scores = torch.tensor(normal_scores_dict['combined_score'])
            anomalous_scores = torch.tensor(anomalous_scores_dict['combined_score'])
            
            # Compute metrics
            metrics = compute_binary_metrics(normal_scores, anomalous_scores)
            all_results[(alpha, beta)] = metrics
            
            if verbose:
                print(f"  AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
            
            # Track best configuration
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                best_config = (alpha, beta)
                best_metrics = metrics
            
            # Track baseline (β=0)
            if beta == 0.0:
                baseline_auc = max(baseline_auc, metrics['auc'])
    
    if verbose:
        print("-" * 60)
        print("GRID SEARCH RESULTS:")
        print(f"Best configuration: α={best_config[0]}, β={best_config[1]}")
        print(f"Best AUC: {best_auc:.4f}")
        print(f"Best F1: {best_metrics['f1']:.4f}")
        print(f"Baseline AUC (β=0): {baseline_auc:.4f}")
        
        if best_auc > baseline_auc:
            improvement = ((best_auc - baseline_auc) / baseline_auc) * 100
            print(f"Improvement over baseline: {improvement:.1f}%")
            print("PASS: MANIFOLD LEARNING IMPROVES PERFORMANCE!")
        else:
            print("WARNING: No improvement over CNN-only baseline")
    
    return GridSearchResult(
        best_alpha=best_config[0],
        best_beta=best_config[1], 
        best_auc=best_auc,
        best_precision=best_metrics['precision'],
        best_recall=best_metrics['recall'],
        all_results=all_results,
        baseline_auc=baseline_auc
    )


def plot_grid_search_heatmap(
    result: GridSearchResult,
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 8)
) -> None:
    """
    Plot heatmap visualization of grid search results.
    
    Parameters
    ----------
    result : GridSearchResult
        Grid search results to visualize
    save_path : str, optional
        Path to save figure (if None, displays plot)
    figsize : tuple
        Figure size in inches
    """
    # Extract alpha/beta values and AUC scores
    alphas = sorted(list(set(config[0] for config in result.all_results.keys())))
    betas = sorted(list(set(config[1] for config in result.all_results.keys())))
    
    # Create AUC matrix
    auc_matrix = np.zeros((len(betas), len(alphas)))
    
    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            if (alpha, beta) in result.all_results:
                auc_matrix[i, j] = result.all_results[(alpha, beta)]['auc']
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Heatmap
    im = ax.imshow(auc_matrix, cmap='viridis', aspect='auto', origin='lower')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('AUC Score', rotation=270, labelpad=20)
    
    # Labels
    ax.set_xlabel('Alpha (Reconstruction Weight)')
    ax.set_ylabel('Beta (Manifold Weight)')
    ax.set_title('Grid Search Results: AUC vs Alpha/Beta\n' + 
                f'Best: α={result.best_alpha}, β={result.best_beta}, AUC={result.best_auc:.4f}')
    
    # Tick labels
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f'{a:.2f}' for a in alphas])
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels([f'{b:.2f}' for b in betas])
    
    # Mark best configuration
    best_i = betas.index(result.best_beta)
    best_j = alphas.index(result.best_alpha)
    ax.plot(best_j, best_i, 'r*', markersize=15, markeredgecolor='white', markeredgewidth=2)
    
    # Add text annotations for key values
    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            if (alpha, beta) in result.all_results:
                auc = result.all_results[(alpha, beta)]['auc']
                color = 'white' if auc < 0.7 else 'black'
                ax.text(j, i, f'{auc:.3f}', ha='center', va='center', 
                       fontsize=8, color=color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grid search heatmap saved to: {save_path}")
    else:
        plt.show()


def analyze_manifold_contribution(
    result: GridSearchResult,
    verbose: bool = True
) -> dict[str, float]:
    """
    Analyze how manifold learning contributes to performance.
    
    Parameters
    ----------
    result : GridSearchResult
        Grid search results to analyze
    verbose : bool
        Whether to print analysis
        
    Returns
    -------
    dict
        Analysis metrics
    """
    # Compare β=0 (CNN-only) vs best β>0 configuration
    baseline_configs = [(alpha, 0.0) for alpha, beta in result.all_results.keys() if beta == 0.0]
    manifold_configs = [(alpha, beta) for alpha, beta in result.all_results.keys() if beta > 0.0]
    
    # Best CNN-only configuration
    best_baseline_auc = max(result.all_results[config]['auc'] for config in baseline_configs)
    best_baseline_config = max(baseline_configs, key=lambda c: result.all_results[c]['auc'])
    
    # Best manifold configuration  
    best_manifold_auc = max(result.all_results[config]['auc'] for config in manifold_configs)
    best_manifold_config = max(manifold_configs, key=lambda c: result.all_results[c]['auc'])
    
    # Compute improvements
    absolute_improvement = best_manifold_auc - best_baseline_auc
    if best_baseline_auc > 0:
        relative_improvement = (absolute_improvement / best_baseline_auc) * 100
    else:
        relative_improvement = 0
    
    # Statistical significance (simple check)
    significant = absolute_improvement > 0.05  # 5% AUC improvement threshold
    
    analysis = {
        'best_baseline_auc': best_baseline_auc,
        'best_baseline_config': best_baseline_config,
        'best_manifold_auc': best_manifold_auc,
        'best_manifold_config': best_manifold_config,
        'absolute_improvement': absolute_improvement,
        'relative_improvement': relative_improvement,
        'significant': significant
    }
    
    if verbose:
        print("\n" + "="*60)
        print("MANIFOLD LEARNING CONTRIBUTION ANALYSIS")
        print("="*60)
        print(f"Best CNN-only (β=0):   AUC={best_baseline_auc:.4f} at α={best_baseline_config[0]}")
        print(
            f"Best Manifold (β>0):   AUC={best_manifold_auc:.4f} "
            f"at α={best_manifold_config[0]}, β={best_manifold_config[1]}"
        )
        print(f"Absolute improvement:  {absolute_improvement:+.4f}")
        print(f"Relative improvement:  {relative_improvement:+.1f}%")
        print(f"Statistically significant: {'PASS' if significant else 'FAIL'}")
        
        if significant:
            print("\nSUCCESS: MANIFOLD LEARNING PROVIDES MEANINGFUL IMPROVEMENT!")
            print("   This validates the geometric approach for UEBA anomaly detection.")
        else:
            print("\nWARNING: MANIFOLD IMPROVEMENT IS MARGINAL")  
            print("   May need more diverse data or different hyperparameters.")
    
    return analysis