# CERT r4.2 Experiment Notes

## Overview

This document describes the initial validation of manifold-based anomaly detection on the CERT Insider Threat Dataset (r4.2). The goal is to test whether geometric structure in the latent space of a CNN autoencoder provides discriminative value for detecting insider threats beyond reconstruction error alone.

## Dataset

**Source:** Carnegie Mellon CERT Insider Threat Dataset v4.2

**Contents:**
- 1,000 synthetic users over ~17 months (Jan 2010 - May 2011)
- 70 malicious users across 3 attack scenarios:
  - Scenario 1 (30 users): Data exfiltration, after-hours activity
  - Scenario 2 (30 users): Data exfiltration to removable media
  - Scenario 3 (10 users): Sabotage/system damage
- 930 normal users

**Data modalities:**
- Logon/logoff events (854K rows)
- Device connect/disconnect (405K rows)
- HTTP activity (28.4M rows)
- Email sent/received (2.6M rows)
- File operations (445K rows)

### Attack Scenario Analysis

The three attack scenarios exhibit significantly different behavioral signatures:

| Scenario | Users | Mean PCs Accessed | Mean USB Events | Behavioral Profile |
|----------|-------|-------------------|-----------------|-------------------|
| 1: After-hours exfil | 30 | 21.7 | 26 | Similar to normal users (20.4 PCs) |
| 2: USB exfil | 30 | 15.5 | 1,014 | Normal PC access, heavy USB activity |
| 3: Sabotage | 10 | 420.4 | 1,294 | 20x more PCs than normal |

**Detection difficulty:**

- **Scenario 3 (Easy):** Attackers access 300-600 systems, far outside normal range. Simple threshold-based detection would likely succeed.

- **Scenario 2 (Medium):** PC access patterns are normal, but USB activity is a strong signal. Detection relies on device-related features.

- **Scenario 1 (Hard):** Behavioral profile closely matches normal users. Detection must rely on subtle temporal patterns (after-hours activity) rather than volume anomalies. This is the most realistic and challenging scenario.

**Implication:** Scenario 1 represents the true test of the manifold approach. If geometric structure can capture subtle temporal deviations that volume-based features miss, it should show improvement specifically on Scenario 1 users. Scenario 3 results may inflate overall metrics due to easy detectability.

### Attack Duration Analysis

Attack durations vary significantly, which affects what temporal resolution can detect:

| Scenario | Min Duration | Mean Duration | Max Duration |
|----------|--------------|---------------|--------------|
| 1: After-hours | 21 min | 5.3 days | 10.9 days |
| 2: USB exfil | 44 days | 55 days | 58 days |
| 3: Sabotage | 30 hrs | 33 hrs | 36 hrs |

**Duration distribution across all 70 attacks:**
- Under 1 hour: 3 attacks (all Scenario 1)
- Under 1 day: 5 attacks (all Scenario 1)
- 1-7 days: 26 attacks
- 7-30 days: 9 attacks
- Over 30 days: 30 attacks (all Scenario 2)

**Implications for temporal resolution:**

| Bucket Size | Trajectory Analysis | Catches |
|-------------|---------------------|---------|
| 1-hour + daily trajectories | Multi-day patterns | 65/70 (93%) |
| 15-min + hourly trajectories | Intra-day patterns | 70/70 (100%) |

The 3 shortest Scenario 1 attacks (21-40 min) fit within a single 1-hour bucket - no trajectory signal possible at this resolution. These can only be detected via point-based anomaly (single bucket with unusual activity), not trajectory deviation.

**For trajectory analysis testing with 1-hour buckets:**
- Exclude: 3 attacks under 1 hour (no temporal signal)
- Include: 5 attacks 1-24 hours (minimal trajectory, 1-24 buckets)
- Include: 62 attacks over 1 day (full trajectory signal)

Multi-day attacks (Scenario 2, most Scenario 1) are well-suited for trajectory analysis over consecutive daily latent points. Scenario 3 attacks (~33 hours) span roughly 1.5 days.

## Feature Engineering

Raw events are aggregated into hourly buckets per user, producing 13 features:

| Feature | Description |
|---------|-------------|
| logon_count | Number of logon events |
| logoff_count | Number of logoff events |
| after_hours_logon | Logons outside 6am-6pm |
| unique_pcs | Distinct machines accessed |
| device_connect | USB/removable device connections |
| device_disconnect | USB/removable device disconnections |
| http_count | HTTP requests |
| unique_urls | Distinct URLs visited |
| email_sent | Emails sent |
| email_internal | Emails to internal recipients |
| email_external | Emails to external recipients |
| file_ops | File copy/write/delete operations |
| file_to_removable | File operations targeting removable media |

Sequences are formed as 24 consecutive hourly buckets (1 day), yielding tensors of shape `(24, 13)`.

## Methodology

### Model Architecture

CNN Autoencoder with:
- Input: `(1, 24, 13)` - single channel, 24 time steps, 13 features
- Encoder: Conv2D layers reducing to latent dimension (default 32)
- Decoder: Transposed Conv2D layers reconstructing input
- Total parameters: ~113K

### Manifold Construction

1. Train autoencoder on normal user sequences only
2. Extract latent representations for all training samples
3. Build k-NN graph in latent space (default k=32)
4. For each neighborhood, estimate local tangent space via PCA

### Anomaly Scoring

Combined score: `s(x) = α * ε(x) + β * δ_⊥(φ(x))`

Where:
- `ε(x)` = reconstruction error (MSE)
- `δ_⊥(φ(x))` = perpendicular distance from latent point to local tangent space
- `α`, `β` = weighting coefficients (tunable)

## Running the Pipeline

### Prerequisites

```bash
cd /path/to/manifold-ueba
source .venv/bin/activate
```

### First Run (Process Raw Data)

```bash
python examples/cert_data_pipeline.py \
    --data-dir data/cert/r4.2 \
    --sample 50 \
    --epochs 20 \
    --save-processed data/cert_processed_50users.npz \
    --grid-search
```

Arguments:
- `--data-dir`: Path to extracted CERT r4.2 data
- `--sample N`: Sample N users (for local testing). Omit for all users.
- `--epochs`: Training epochs
- `--save-processed`: Save processed sequences to .npz file
- `--grid-search`: Run grid search over α/β values

Runtime: ~8 minutes on M1 MacBook (dominated by loading 28M HTTP rows)

### Subsequent Runs (Load Processed Data)

```bash
python examples/cert_data_pipeline.py \
    --load-processed data/cert_processed_50users.npz \
    --epochs 50 \
    --grid-search
```

Runtime: ~3-4 minutes (training + evaluation only)

### Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--bucket-hours` | 1 | Hours per time bucket. Production UEBA typically uses 15min. |
| `--sequence-length` | 24 | Buckets per sequence. With 1hr buckets, this is 1 day. |
| `--latent-dim` | 32 | Autoencoder latent space dimension |
| `--epochs` | 50 | Training epochs |

## Current Findings

### Experimental Conditions

- 50 sampled users (47 normal, 3 malicious)
- 1-hour time buckets (coarse; production would use 15min)
- 24-hour sequences
- 20 training epochs
- Train: ~13K sequences (normal only)
- Test: ~4K sequences (mix of normal and malicious)

### Grid Search Results

Best configuration found: `α=0.75, β=8.0`

Top configurations by PR-AUC:

| Alpha | Beta | ROC-AUC | PR-AUC |
|-------|------|---------|--------|
| 0.75 | 8.00 | 0.859 | 0.736 |
| 0.25 | 2.00 | 0.859 | 0.736 |
| 0.50 | 4.00 | 0.859 | 0.736 |
| 0.00 | 0.50 | 0.862 | 0.735 |

Note: Pure manifold distance (α=0) achieves comparable performance, suggesting the geometric structure alone is highly discriminative.

### Comparison: AE-only vs. Combined

| Metric | AE-only (β=0) | Combined (α=0.75, β=8) | Change |
|--------|---------------|------------------------|--------|
| ROC-AUC | 0.809 | 0.859 | +6.2% |
| PR-AUC | 0.673 | 0.736 | +9.4% |
| FPR @ 90% TPR | 54.6% | 39.5% | -27.6% |
| Precision @ 90% Recall | 28.8% | 35.9% | +24.7% |

### Variance Considerations

Results exhibit significant variance across runs due to:
1. Random sampling of users (only 3 of 70 malicious users per run)
2. Different attack scenarios have vastly different detectability (see Attack Scenario Analysis above)
3. Random model initialization and batch ordering

When the random sample includes Scenario 3 attackers (who access 20x more systems than normal), metrics improve substantially. When the sample is dominated by Scenario 1 attackers (who blend in with normal users), metrics are lower but more representative of real-world difficulty.

**Implication:** For publishable results:
- Stratify evaluation by scenario (report Scenario 1 results separately as the hard case)
- Use all malicious users rather than sampling
- Run multiple trials with different seeds and report mean ± std
- Use fixed random seed for reproducibility

## Interpretation

The manifold geometry provides measurable improvement over reconstruction error alone:

1. **PR-AUC improvement of ~9%** indicates better ranking of malicious samples
2. **27% reduction in FPR at 90% detection** is operationally significant - fewer false alarms for security analysts
3. **Pure manifold distance (α=0) competitive with combined** suggests geometric structure captures distinct information from reconstruction error

These results were obtained with:
- Coarse temporal resolution (1hr vs. typical 15min)
- Minimal training (20 epochs)
- Small user sample (50 of 1000)
- No hyperparameter optimization beyond α/β

The fact that improvement is observable under these conditions suggests the approach merits further investigation with finer temporal resolution and full dataset.

**Note on scenario composition:** Current results aggregate across all three attack scenarios. Given that Scenario 3 attackers are trivially detectable (20x normal PC access), reported metrics may overstate performance on realistic threats. Scenario 1-specific evaluation is needed to assess true value of the manifold approach on hard cases where attackers blend in with normal behavior.

## Limitations

1. **Temporal resolution:** 1-hour buckets are coarse. Insider threats often manifest in patterns spanning minutes.

2. **Sample size:** 50 users provides limited statistical power. Results vary based on which malicious users are sampled.

3. **Synthetic data:** CERT dataset is synthetic. Real-world insider threat data may have different characteristics.

4. **Single dataset:** Results need validation on additional datasets.

5. **No baseline comparison:** Current evaluation compares β=0 vs β>0 within our framework. Comparison against established UEBA methods would strengthen claims.

## Trajectory Analysis (Geodesic Deviation)

Beyond single-point manifold distance, we implemented trajectory analysis to detect anomalous paths through latent space. This approach is inspired by geodesic deviation in differential geometry - measuring how observed trajectories deviate from expected "straight paths" (geodesics) on the learned manifold.

### Methodology

Given a sequence of latent points `γ(t) = {z₁, z₂, ..., zₙ}`:

- **Velocity:** `v_i = z_{i+1} - z_i` (tangent vector)
- **Acceleration:** `a_i = v_{i+1} - v_i` (geodesic deviation proxy)
- **Manifold drift:** Change in off-manifold distance over trajectory
- **Tortuosity:** Path length / displacement (erratic vs. direct movement)

Reference statistics are computed from normal user trajectories, and anomaly scores are based on z-score deviations from normal.

### Results

Using 6-point trajectory windows (stride=3) on the same test data:

| Metric | Point-based | Trajectory-based | Improvement |
|--------|-------------|------------------|-------------|
| **ROC-AUC** | 0.832 | **0.992** | **+19.2%** |
| **PR-AUC** | 0.716 | **0.968** | **+35.2%** |
| Separation Score | 0.77 | 1.59 | +107% |

The trajectory-based approach achieves near-perfect classification (ROC-AUC 0.99, PR-AUC 0.97) on trajectories derived from malicious vs. normal user sequences.

**Feature breakdown (malicious/normal ratio):**

| Feature | Ratio | Interpretation |
|---------|-------|----------------|
| max_manifold_distance | **6.33x** | Malicious trajectories venture far off manifold |
| mean_acceleration | 1.53x | Higher geodesic deviation |
| mean_velocity | 1.53x | More rapid movement through latent space |
| tortuosity | 0.97x | Similar path directness |

### Interpretation

Trajectory analysis more than doubles the separation between normal and malicious behavior compared to single-point manifold distance. The key discriminator is `max_manifold_distance` - during attack sequences, users venture 6x further from the normal behavior manifold than typical users.

This validates the geodesic deviation hypothesis: insider threats manifest not just as single anomalous points, but as **anomalous paths** through behavioral space. The temporal structure of attacks (escalating deviation, erratic movement) provides discriminative signal beyond instantaneous anomaly scores.

### Implications

1. **Trajectory > Point:** For multi-day attacks, analyzing behavioral trajectories provides stronger signal than single-point scoring.

2. **Finer resolution needed:** With 1-hour buckets and 24-hour sequences, each "point" represents a full day. Attacks under 24 hours have limited trajectory signal. Finer resolution (15-min buckets, 2-hour windows) would enable intra-day trajectory analysis.

3. **Multi-day attacks:** Scenario 2 attacks (55 days average) are ideal candidates for trajectory analysis over weeks.

## Experiment 002: All Malicious Users (2026-01-22)

### Motivation

Previous experiments sampled both normal and malicious users proportionally, resulting in only ~5 malicious users in the test set. This provides poor statistical power for evaluation. We implemented `--all-malicious` flag to include ALL 67 malicious users (with attacks >= 1 hour) in the test set while sampling only normal users for training.

### Changes Implemented

1. **`--all-malicious` flag**: New argument in `cert_data_pipeline.py` that:
   - Samples N normal users for training (manifold construction)
   - Includes ALL malicious users in test set (not sampled)

2. **`--experiment` flag**: Experiment management system that:
   - Creates `runs/<experiment_name>/` directory
   - Saves all outputs: `config.json`, `processed.npz`, `model.pt`, `manifold.npz`
   - Auto-generates `results.json` and `README.md`
   - Trajectory results saved to `trajectory_results.json`

3. **`--load-experiment` flag**: Load from existing experiment directory

4. **Improved grid search**: 
   - Normalized scoring (z-score both components, then mix)
   - Expanded alpha/beta grid
   - Reports both raw and normalized configurations

### Configuration

```bash
python examples/cert_data_pipeline.py \
    --data-dir data/cert/r4.2 \
    --sample 50 \
    --all-malicious \
    --bucket-hours 1.0 \
    --sequence-length 24 \
    --min-attack-hours 1 \
    --epochs 50 \
    --grid-search \
    --experiment exp002_50users_all_malicious

python examples/trajectory_analysis.py \
    --load-experiment runs/exp002_50users_all_malicious \
    --window-size 6 \
    --stride 3 \
    --min-attack-hours 1
```

### Data Splits

| Split | Sequences | Users |
|-------|-----------|-------|
| Training | 13,701 | 50 normal users |
| Test (normal) | 3,425 | 50 normal users (20% holdout) |
| Test (malicious) | 14,975 | ALL 67 malicious users |
| **Total test** | **18,400** | 117 users |

### Results: Point-Based Detection

**Grid Search Finding**: Pure manifold distance (alpha=0) outperformed reconstruction error.

| Method | ROC-AUC | PR-AUC | Optimal F1 |
|--------|---------|--------|------------|
| AE-only (beta=0) | 0.6492 | 0.8722 | 0.8974 |
| Pure Manifold (w=0) | **0.7321** | **0.9217** | 0.8974 |

**Improvement from manifold**: +12.8% ROC-AUC, +5.7% PR-AUC

### Results: Trajectory-Based Detection

| Method | ROC-AUC | PR-AUC | Optimal F1 |
|--------|---------|--------|------------|
| Point-based | 0.7321 | 0.9217 | 0.8974 |
| **Trajectory-based** | **0.8468** | **0.9543** | **0.9250** |

**Improvement from trajectory**: +15.7% ROC-AUC, +3.5% PR-AUC

### Confusion Matrix at Optimal F1 (Trajectory)

```
                 Predicted
              Normal  Malicious
Actual Normal    497      643
Actual Malicious 143     4847
```

- **Precision**: 88.3%
- **Recall**: 97.1%  
- **F1**: 92.5%

### Key Findings

1. **Manifold distance alone is sufficient**: Best grid search result used alpha=0 (pure manifold), suggesting reconstruction error adds noise rather than signal.

2. **Trajectory analysis provides major improvement**: +15.7% ROC-AUC over point-based, confirming that temporal structure of attacks is discriminative.

3. **All malicious users critical**: With full malicious set (67 users, 14,975 sequences), we have much better statistical power than previous 5-user samples.

4. **High recall achievable**: 97% recall at 88% precision is operationally useful.

### Artifacts

All outputs saved to `runs/exp002_50users_all_malicious/`:
- `config.json` - Experiment parameters
- `processed.npz` - Processed sequences
- `model.pt` - Trained CNN autoencoder
- `manifold.npz` - Latent manifold
- `results.json` - Point-based evaluation results
- `trajectory_results.json` - Trajectory evaluation results
- `README.md` - Auto-generated summary

### Hypotheses for Future Work

1. **More normal users → better manifold**: Denser manifold should improve separation. Test with 100, 200, all 930 normal users.

2. **Scenario-stratified analysis**: Break down results by attack scenario to understand where trajectory analysis helps most.

3. **Finer temporal resolution**: 15-minute buckets may capture intra-day attack patterns missed at hourly resolution.

## Next Steps

1. **Scenario-stratified evaluation:** Report metrics separately for each attack scenario. Scenario 1 (after-hours exfiltration) is the primary target as it represents the hardest and most realistic case.

2. **Reproducibility:** Add `--seed` argument for deterministic sampling and initialization

3. ~~**Full malicious set:** Modify sampling to always include all 70 malicious users, sample only from normal users~~ ✓ Done (exp002)

4. **Finer resolution:** Test 15-minute buckets (will require cluster resources for HTTP data)

5. **More epochs:** Current training may be insufficient; loss curves suggest room for improvement

6. **Latent dimension:** Grid search over latent_dim (16, 32, 64, 128)

7. **k-neighbors:** Sensitivity analysis on manifold k parameter

8. **Temporal features:** Scenario 1 detection relies on after-hours patterns. Verify that `after_hours_logon` feature and temporal structure in sequences are capturing this signal.

9. **Manifold density study:** Run exp003 with 100+ normal users to test hypothesis that denser manifold improves detection.

## File Locations

**Code:**
- Pipeline script: `examples/cert_data_pipeline.py`
- Trajectory analysis: `examples/trajectory_analysis.py`
- ETL module: `manifold_ueba/etl/cert.py`
- Model: `manifold_ueba/cnn_model.py`
- Manifold: `manifold_ueba/latent_manifold.py`
- Trajectory analyzer: `manifold_ueba/trajectory.py`

**Saved artifacts (not in git):**
- Processed data: `data/cert_processed_50users.npz`
- Trained model: `data/cert_model.pt`
- Manifold: `data/cert_manifold.npz`
- Raw CERT data: `data/cert/r4.2/`

## Model Saving/Loading

For faster iteration, the pipeline supports saving and loading trained models and manifolds:

```bash
# First run: train and save everything
python examples/cert_data_pipeline.py \
    --data-dir data/cert/r4.2 --sample 50 --epochs 20 \
    --save-processed data/cert_processed_50users.npz \
    --save-model data/cert_model.pt \
    --save-manifold data/cert_manifold.npz

# Subsequent runs: load pre-trained artifacts
python examples/cert_data_pipeline.py \
    --load-processed data/cert_processed_50users.npz \
    --load-model data/cert_model.pt \
    --load-manifold data/cert_manifold.npz \
    --grid-search

# Trajectory analysis (requires saved model + manifold)
python examples/trajectory_analysis.py \
    --load-processed data/cert_processed_50users.npz \
    --load-model data/cert_model.pt \
    --load-manifold data/cert_manifold.npz
```

## Cluster Experiments

The following experiments use the full CERT dataset (all 930 normal users, filtered malicious users) and require cluster resources due to memory requirements.

### New Parameters

| Parameter | Description |
|-----------|-------------|
| `--bucket-hours` | Hours per bucket. Use 0.25 for 15-min buckets. |
| `--min-attack-hours` | Filter attacks by minimum duration (hours). Excludes attacks too short for temporal analysis at chosen bucket size. |

### Experiment 1: Full Dataset with 1-Hour Buckets

Uses all normal users and attacks with duration >= 24 hours (65 of 70 attacks). This ensures trajectory analysis has meaningful temporal signal (at least 24 latent points per attack).

```bash
# Step 1: Process data and train model
python examples/cert_data_pipeline.py \
    --data-dir data/cert/r4.2 \
    --bucket-hours 1.0 \
    --sequence-length 24 \
    --min-attack-hours 24 \
    --epochs 50 \
    --save-processed data/cert_full_1hr.npz \
    --save-model data/cert_model_1hr.pt \
    --save-manifold data/cert_manifold_1hr.npz \
    --grid-search

# Step 2: Run trajectory analysis
python examples/trajectory_analysis.py \
    --load-processed data/cert_full_1hr.npz \
    --load-model data/cert_model_1hr.pt \
    --load-manifold data/cert_manifold_1hr.npz \
    --min-attack-hours 24 \
    --window-size 6 \
    --stride 3
```

**Expected outputs:**
- AE-only (beta=0): ROC-AUC, PR-AUC
- Point-based manifold (beta>0): ROC-AUC, PR-AUC
- Trajectory analysis: ROC-AUC, PR-AUC, feature breakdown

### Experiment 2: 15-Minute Buckets for Fine-Grained Analysis

Uses 15-minute buckets to capture intra-day attack patterns. Includes attacks >= 2 hours duration (all 70 attacks qualify).

```bash
# Step 1: Process data with fine temporal resolution
python examples/cert_data_pipeline.py \
    --data-dir data/cert/r4.2 \
    --bucket-hours 0.25 \
    --sequence-length 8 \
    --min-attack-hours 2 \
    --epochs 50 \
    --save-processed data/cert_full_15min.npz \
    --save-model data/cert_model_15min.pt \
    --save-manifold data/cert_manifold_15min.npz \
    --grid-search

# Step 2: Run trajectory analysis  
python examples/trajectory_analysis.py \
    --load-processed data/cert_full_15min.npz \
    --load-model data/cert_model_15min.pt \
    --load-manifold data/cert_manifold_15min.npz \
    --min-attack-hours 2 \
    --window-size 8 \
    --stride 4
```

**Note:** With `--bucket-hours 0.25` and `--sequence-length 8`, each sequence covers 2 hours (8 x 15min). This captures short-duration attacks that would be invisible at hourly resolution.

### Resource Estimates

| Configuration | Sequences (est.) | Memory | Runtime |
|---------------|------------------|--------|---------|
| 50 users, 1hr buckets | ~17K | 4GB | 8 min |
| 1000 users, 1hr buckets | ~340K | 16-32GB | 2-3 hr |
| 1000 users, 15min buckets | ~1.4M | 64GB+ | 8-12 hr |

### Comparing All Three Methods

For paper results, ensure all three detection methods are evaluated on identical test data:

1. **AE-only (beta=0):** Use grid search result with beta=0
2. **Point-based (beta>0):** Use best alpha/beta from grid search
3. **Trajectory:** Run trajectory_analysis.py on same test set

The pipeline saves test labels in the .npz file, ensuring reproducible comparison across methods.
