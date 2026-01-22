#!/usr/bin/env python3
"""
Test CNN + Manifold Architecture for UEBA

This script validates the new CNN-based manifold learning approach, 
following the successful "data-space-time" concept from gravitational wave detection.

Key test: Does β > 0 contribute when we treat (time, features) as spatial coordinates?
"""

import os
import sys

import numpy as np
import torch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manifold_ueba import (
    UEBACNNAutoencoder,
    UEBALatentManifold,
    UEBAManifoldConfig,
    UEBAManifoldScorer,
    UEBAManifoldScorerConfig,
    generate_diverse_sequence,
    prepare_ueba_sequences_for_cnn,
)


def generate_test_data(n_normal=50, n_anomalous=30):
    """Generate test data for CNN + manifold validation."""
    print("Generating UEBA behavioral sequences for CNN processing...")
    
    # Generate diverse normal sequences
    normal_sequences = []
    archetypes = ["developer", "executive", "support", "analyst"]
    
    np.random.seed(42)
    for i in range(n_normal):
        archetype = archetypes[i % len(archetypes)]
        sequence = generate_diverse_sequence(
            archetype=archetype,
            seq_len=24, 
            anomalous=False,
            noise_level=0.2
        )
        normal_sequences.append(sequence)
    
    # Generate anomalous sequences
    anomalous_sequences = []
    for i in range(n_anomalous):
        archetype = archetypes[i % len(archetypes)]
        sequence = generate_diverse_sequence(
            archetype=archetype,
            seq_len=24,
            anomalous=True,  # Coordinated attack patterns
            noise_level=0.2
        )
        anomalous_sequences.append(sequence)
    
    # Convert to tensors and prepare for CNN
    normal_tensor = torch.FloatTensor(np.stack(normal_sequences))
    anomalous_tensor = torch.FloatTensor(np.stack(anomalous_sequences))
    
    # Prepare for CNN (add channel dim and normalize)
    normal_cnn = prepare_ueba_sequences_for_cnn(normal_tensor)
    anomalous_cnn = prepare_ueba_sequences_for_cnn(anomalous_tensor)
    
    print(f"  Normal sequences: {normal_cnn.shape}")
    print(f"  Anomalous sequences: {anomalous_cnn.shape}")
    
    return normal_cnn, anomalous_cnn


def train_cnn_quickly(model, normal_data, epochs=50):
    """Quick training of CNN autoencoder on normal data."""
    print("Training CNN autoencoder on normal behavioral patterns...")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        # Simple batch training 
        batch_size = 10
        for i in range(0, len(normal_data), batch_size):
            batch = normal_data[i:i+batch_size]
            
            optimizer.zero_grad()
            
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            avg_loss = total_loss / (len(normal_data) // batch_size + 1)
            print(f"  Epoch {epoch:2d}: Loss = {avg_loss:.6f}")
    
    model.eval()
    print("CNN training complete")
    return model


def test_cnn_autoencoder(normal_data):
    """Test CNN autoencoder on UEBA data with training."""
    print("\n" + "="*50)
    print("TESTING CNN AUTOENCODER")
    print("="*50)
    
    # Create model
    model = UEBACNNAutoencoder(
        time_steps=24,
        n_features=13,
        latent_dim=32
    )
    
    print(f"Model info: {model.get_model_info()}")
    
    # Train the model on normal data
    model = train_cnn_quickly(model, normal_data, epochs=50)
    
    # Test forward pass
    test_input = normal_data[:5]  # Use actual data for testing
    reconstructed, latent = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Latent shape: {latent.shape}")
    
    # Test individual encode/decode
    encoded = model.encode(test_input)
    decoded = model.decode(encoded)
    
    print(f"Encoded shape: {encoded.shape}")
    print(f"Decoded shape: {decoded.shape}")
    
    # Check reconstruction quality after training
    recon_error = torch.mean((test_input - reconstructed) ** 2).item()
    print(f"Mean reconstruction error: {recon_error:.6f}")
    
    # Check latent diversity (important for manifold learning)
    latent_std = torch.std(latent, dim=0).mean().item()
    print(f"Latent diversity (std): {latent_std:.6f}")
    
    assert reconstructed.shape == test_input.shape
    assert latent.shape[0] == 5
    assert latent.shape[1] == 32
    
    print("PASS: CNN autoencoder training and forward pass successful")
    return model


def test_latent_manifold(cnn_model, normal_data):
    """Test latent manifold construction and scoring."""
    print("\n" + "="*50)
    print("TESTING LATENT MANIFOLD")
    print("="*50)
    
    # Extract latents from normal data  
    cnn_model.eval()
    with torch.no_grad():
        _, latents = cnn_model(normal_data)
        latents_np = latents.cpu().numpy()
    
    print(f"Training latents shape: {latents_np.shape}")
    
    # Build manifold
    config = UEBAManifoldConfig(k_neighbors=16, tangent_dim=8)
    manifold = UEBALatentManifold(latents_np, config)
    
    manifold_info = manifold.get_manifold_info()
    print(f"Manifold info: {manifold_info}")
    
    # Validate manifold quality
    quality = manifold.validate_manifold_quality()
    print(f"Manifold quality: {quality}")
    
    # Test scoring on sample points
    test_point = latents_np[0]
    off_manifold_dist = manifold.normal_deviation(test_point)
    density_score = manifold.density_score(test_point)
    
    print(f"Sample point off-manifold distance: {off_manifold_dist:.6f}")
    print(f"Sample point density score: {density_score:.6f}")
    
    assert off_manifold_dist >= 0, "Off-manifold distance should be non-negative"
    assert density_score >= 0, "Density score should be non-negative"
    
    print("PASS: Latent manifold construction successful")
    return manifold


def test_manifold_scorer(cnn_model, manifold, normal_data, anomalous_data):
    """Test manifold-based scoring with different α/β combinations."""
    print("\n" + "="*50)
    print("TESTING MANIFOLD SCORER")
    print("="*50)
    
    # Test different configurations
    configs = [
        ("CNN Only (β=0)", UEBAManifoldScorerConfig(mode="cnn_only", alpha_cnn=1.0)),
        ("Manifold Only", UEBAManifoldScorerConfig(mode="manifold_only", beta_manifold=1.0)),
        (
            "CNN + Manifold (β=0.5)", 
            UEBAManifoldScorerConfig(mode="cnn_plus_manifold", alpha_cnn=1.0, beta_manifold=0.5)
        ),
        (
            "CNN + Manifold (β=1.0)", 
            UEBAManifoldScorerConfig(mode="cnn_plus_manifold", alpha_cnn=1.0, beta_manifold=1.0)
        ),
        (
            "CNN + Manifold (β=2.0)", 
            UEBAManifoldScorerConfig(mode="cnn_plus_manifold", alpha_cnn=1.0, beta_manifold=2.0)
        ),
    ]
    
    results = {}
    
    for config_name, config in configs:
        print(f"\nTesting: {config_name}")
        
        scorer = UEBAManifoldScorer(manifold, config)
        
        # Score normal data
        normal_scores = scorer.score_batch(cnn_model, normal_data)
        
        # Score anomalous data  
        anomalous_scores = scorer.score_batch(cnn_model, anomalous_data)
        
        # Compute separation
        normal_mean = np.mean(normal_scores['combined_score'])
        anomalous_mean = np.mean(anomalous_scores['combined_score'])
        normal_std = np.std(normal_scores['combined_score'])
        anomalous_std = np.std(anomalous_scores['combined_score'])
        
        separation = (anomalous_mean - normal_mean) / (normal_std + anomalous_std + 1e-8)
        
        results[config_name] = {
            'normal_mean': normal_mean,
            'anomalous_mean': anomalous_mean,
            'separation': separation,
            'beta': config.beta_manifold,
            'scores': (normal_scores, anomalous_scores)
        }
        
        print(f"  Normal mean: {normal_mean:.4f} ± {normal_std:.4f}")
        print(f"  Anomalous mean: {anomalous_mean:.4f} ± {anomalous_std:.4f}")
        print(f"  Separation: {separation:.4f}")
        
        # Component breakdown
        if 'reconstruction_error' in normal_scores:
            recon_mean = np.mean(normal_scores['reconstruction_error'])
            print(f"  Reconstruction component: {recon_mean:.4f}")
        
        if 'off_manifold_distance' in normal_scores:
            manifold_mean = np.mean(normal_scores['off_manifold_distance'])
            manifold_std = np.std(normal_scores['off_manifold_distance'])
            print(f"  Manifold component: {manifold_mean:.4f} ± {manifold_std:.4f}")
    
    return results


def analyze_results(results):
    """Analyze whether β > 0 provides improvement."""
    print("\n" + "="*50)
    print("RESULTS ANALYSIS")
    print("="*50)
    
    baseline_separation = results["CNN Only (β=0)"]['separation']
    
    print(f"Baseline (β=0) separation: {baseline_separation:.4f}")
    
    # Check β > 0 configurations
    beta_configs = [k for k in results.keys() if "β=" in k and "β=0" not in k]
    
    improvements = []
    for config_name in beta_configs:
        beta_separation = results[config_name]['separation']
        improvement = ((beta_separation - baseline_separation) / abs(baseline_separation)) * 100
        improvements.append(improvement)
        
        print(f"{config_name}: {beta_separation:.4f} ({improvement:+.1f}%)")
    
    print("\n" + "-"*50)
    print("CRITICAL TEST RESULTS:")
    
    best_improvement = max(improvements) if improvements else 0
    best_config = None
    
    for config_name in beta_configs:
        beta_separation = results[config_name]['separation']
        improvement = ((beta_separation - baseline_separation) / abs(baseline_separation)) * 100
        if abs(improvement - best_improvement) < 1e-6:
            best_config = config_name
            break
    
    print(f"Best β configuration: {best_config}")
    print(f"Best improvement: {best_improvement:+.1f}%")
    
    # Check if manifold distances have variance
    manifold_scores = results["CNN + Manifold (β=1.0)"]['scores'][0]['off_manifold_distance']
    manifold_variance = np.var(manifold_scores)
    
    print(f"Manifold distance variance: {manifold_variance:.6f}")
    
    if manifold_variance > 1e-6:
        print("PASS: Manifold distances have variance (geometric structure detected)")
    else:
        print("FAIL: Manifold distances have no variance (degenerate manifold)")
    
    if best_improvement > 5:  # 5% improvement threshold
        print("PASS: β > 0 provides meaningful improvement!")
        print("      CNN + Manifold architecture is working correctly")
    elif best_improvement > 0:
        print("MARGINAL: β > 0 provides some improvement")
        print("          May need better data or hyperparameter tuning")
    else:
        print("FAIL: β > 0 provides no improvement")
        print("      Architecture may need fundamental changes")
    
    return best_improvement


def main():
    """Run complete CNN + manifold validation test."""
    print("CNN + MANIFOLD ARCHITECTURE VALIDATION")
    print("Testing 'data-space-time' approach for UEBA")
    print("-" * 60)
    
    try:
        # Step 1: Generate test data
        normal_data, anomalous_data = generate_test_data()
        
        # Step 2: Test CNN autoencoder (with training)
        cnn_model = test_cnn_autoencoder(normal_data)
        
        # Step 3: Test latent manifold
        manifold = test_latent_manifold(cnn_model, normal_data)
        
        # Step 4: Test manifold scorer
        results = test_manifold_scorer(cnn_model, manifold, normal_data, anomalous_data)
        
        # Step 5: Analyze results
        improvement = analyze_results(results)
        
        print("\n" + "="*60)
        print("VALIDATION COMPLETE")
        print("="*60)
        
        if improvement > 5:
            print("SUCCESS: CNN + Manifold architecture is functional!")
            print("Ready for comprehensive evaluation and real data testing.")
        else:
            print("WARNING: CNN + Manifold architecture needs improvement")
            print("Consider data quality, hyperparameters, or architectural changes.")
            
        return improvement > 0
        
    except Exception as e:
        print(f"ERROR in validation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()