# Window Size Ablation Study

**Branch:** `feature/window-size-ablation`  
**Purpose:** Investigate the effect of temporal window size (12h, 24h, 48h) on UEBA anomaly detection performance

---

## Overview

This study evaluates three temporal window sizes for behavioral anomaly detection on the CERT r4.2 insider threat dataset:

- **12-hour windows:** Higher temporal resolution, more training data
- **24-hour windows:** Balanced approach (baseline)
- **48-hour windows:** Longer context, fewer but richer windows

All experiments use the **same architecture and hyperparameters**, varying only the window size.

---

## Results Summary

| Window Size | Mask PR-AUC | Combined PR-AUC | ROC-AUC | vs 24h |
|-------------|-------------|-----------------|---------|--------|
| **12-hour** | 0.277 | 0.282 | 0.736 | -61% |
| **24-hour**  | **0.714** | 0.716 | 0.849 | baseline |
| **48-hour** | 0.682 | 0.682 | 0.879 | -4.5% |

**Finding:** 24-hour windows are optimal for this task.

---

## Prerequisites

1. **Python 3.8+** with dependencies:
   ```bash
   pip install torch numpy pandas scikit-learn scipy matplotlib
   ```

2. **CERT r4.2 Dataset:**
   - Download from: https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
   - Extract to: `data/cert/r4.2/`
   - Required files:
     - `logon.csv`
     - `device.csv`
     - `http.csv`
     - `file.csv`
     - `email.csv`
     - `insiders.csv`

---

## Experiments

### **Experiment 1: 12-Hour Windows**

```bash
python examples/exp005_fixed_window_pipeline.py \
    --experiment exp013_12hour_lambda002_temp002 \
    --data-dir data/cert/r4.2 \
    --window-hours 12 \
    --epochs 50 \
    --batch-size 32 \
    --latent-dim 32 \
    --lr 0.001 \
    --use-mask-value \
    --lambda-value 0.02 \
    --use-temporal-reg \
    --lambda-temporal 0.02 \
    --grid-search
```

**Expected Results:**
- Mask PR-AUC: ~0.277
- Training time: ~15 minutes

---

### **Experiment 2: 24-Hour Windows (Baseline)** 

```bash
python examples/exp005_fixed_window_pipeline.py \
    --experiment exp012_lambda002_temp002 \
    --data-dir data/cert/r4.2 \
    --window-hours 24 \
    --epochs 50 \
    --batch-size 32 \
    --latent-dim 32 \
    --lr 0.001 \
    --use-mask-value \
    --lambda-value 0.02 \
    --use-temporal-reg \
    --lambda-temporal 0.02 \
    --grid-search
```

**Expected Results:**
- Mask PR-AUC: ~0.714
- Training time: ~10 minutes

---

### **Experiment 3: 48-Hour Windows**

```bash
python examples/exp005_fixed_window_pipeline.py \
    --experiment exp014_48hour_lambda002_temp002 \
    --data-dir data/cert/r4.2 \
    --window-hours 48 \
    --epochs 50 \
    --batch-size 32 \
    --latent-dim 32 \
    --lr 0.001 \
    --use-mask-value \
    --lambda-value 0.02 \
    --use-temporal-reg \
    --lambda-temporal 0.02 \
    --grid-search
```

**Expected Results:**
- Mask PR-AUC: ~0.682
- Training time: ~8 minutes

---

## Reproducing Results

### Quick Start

Run all three experiments sequentially:

```bash
# 12-hour
python examples/exp005_fixed_window_pipeline.py \
    --experiment exp013_12hour_lambda002_temp002 \
    --data-dir data/cert/r4.2 --window-hours 12 \
    --epochs 50 --batch-size 32 --latent-dim 32 --lr 0.001 \
    --use-mask-value --lambda-value 0.02 \
    --use-temporal-reg --lambda-temporal 0.02 --grid-search

# 24-hour
python examples/exp005_fixed_window_pipeline.py \
    --experiment exp012_lambda002_temp002 \
    --data-dir data/cert/r4.2 --window-hours 24 \
    --epochs 50 --batch-size 32 --latent-dim 32 --lr 0.001 \
    --use-mask-value --lambda-value 0.02 \
    --use-temporal-reg --lambda-temporal 0.02 --grid-search

# 48-hour
python examples/exp005_fixed_window_pipeline.py \
    --experiment exp014_48hour_lambda002_temp002 \
    --data-dir data/cert/r4.2 --window-hours 48 \
    --epochs 50 --batch-size 32 --latent-dim 32 --lr 0.001 \
    --use-mask-value --lambda-value 0.02 \
    --use-temporal-reg --lambda-temporal 0.02 --grid-search
```

**Total runtime:** ~35 minutes (sequential)

---

## Understanding Results

### Output Files (per experiment)

Each experiment creates a directory: `runs/expXXX_*hour_*/` containing:

**Results (included in repo):**
- `config.json` - Experiment configuration
- `window_level_results.json` - Window detection metrics
- `trajectory_level_results.json` - Trajectory detection metrics
- `pr_curve_by_scenario.png` - Precision-recall curves by attack scenario

**Data (excluded, too large):**
- `autoencoder.pt` - Trained model
- `manifold.npz` - Latent manifold
- `window_level_scores.npz` - Anomaly scores
- `trajectory_level_scores.npz` - Trajectory scores

### Key Metrics

**Window-Level Detection:**
- **Mask PR-AUC:** Primary metric (precision-recall area under curve)
- **ROC-AUC:** Ranking quality
- **Precision/Recall:** Operating point trade-offs

**Trajectory-Level Detection:**
- **Trajectory PR-AUC:** Sequential pattern detection
- **Geodesic Deviation:** Manifold-based anomaly scoring

---

## Comparison Plots

Pre-generated comparison plots are included:

1. **Individual plots per window size:**
   - `runs/exp013_12hour_lambda002_temp002/pr_curve_by_scenario.png`
   - `runs/exp012_lambda002_temp002/pr_curve_by_scenario.png`
   - `runs/exp014_48hour_lambda002_temp002/pr_curve_by_scenario.png`

2. **Overall comparison:**
   - `runs/window_size_comparison_overall.png` (3 curves: 12h, 24h, 48h)
   - `runs/window_size_comparison_all_scenarios.png` (6 curves: S1 & S3 for each)

---

## Architecture

**CNN Autoencoder:**
- **Input:** (T, F) where T = time buckets, F = 12 behavioral features
- **Encoding:** TÃ—F  (T/2)Ã—(F/2)  (T/4)Ã—(F/4)  latent_dim=32
- **Decoding:** Symmetric upsampling back to TÃ—F
- **Loss:** Dual-channel (mask + value) with temporal smoothness regularization

**Hyperparameters (fixed across all experiments):**
- Î»_value = 0.02 (value loss weight)
- Î»_temporal = 0.02 (temporal consistency weight)
- pos_weight = ~10.4 (class imbalance correction, computed per experiment)
- Epochs = 50
- Batch size = 32
- Learning rate = 0.001

---

## Key Findings

### 1. **24-hour windows are optimal**
- Best PR-AUC (0.714)
- Balance between context and granularity
- Sufficient to capture attack patterns (1-7 day duration)

### 2. **12-hour windows are too short**
- 61% worse performance
- Insufficient context
- Too much noise, not enough signal

### 3. **48-hour windows are slightly worse**
- Only 4.5% drop (close to 24h)
- Better ROC-AUC but worse PR-AUC
- Coarser granularity misses attack timing
- **Significantly worse** for Scenario 3 (removable media exfiltration)

### 4. **Scenario-specific insights**
- **Scenario 1 (Logon+Device):** Stable across 24h and 48h
- **Scenario 3 (Logon+Removable):** Needs finer temporal resolution (24h >> 48h)

---

## Reproducibility Notes

### Data Split
- **Chronological split:** Training uses temporally earlier normal data
- **Deterministic:** No randomness in train/test split
- **Buffer:** 7-day temporal separation around attacks

### Random Seeds
- PyTorch: seed=42
- NumPy: seed=42
- Data loader: deterministic ordering with seed

### Expected Variation
- Results should be within Â±2% PR-AUC due to:
  - Hardware differences (CPU vs GPU)
  - Minor numerical precision differences
  - Library version differences

---

## Citation

If you use this work, please cite:

```
[Paper details TBD]
```

---

## File Structure

```
manifold-ueba/
 README.md                              # This file
 examples/
‚    exp005_fixed_window_pipeline.py   # Main experiment script
 manifold_ueba/
‚    __init__.py
‚    cnn_model.py                      # CNN autoencoder
‚    data.py                           # Dataset classes
‚    latent_manifold.py                # Manifold construction
‚    trajectory.py                     # Trajectory scoring
‚    scoring.py                        # Anomaly scoring
‚    etl/
‚        __init__.py
‚        cert_fixed_window.py          # Data loader
 runs/
     exp012_lambda002_temp002/         # 24h results 
     exp013_12hour_lambda002_temp002/  # 12h results
     exp014_48hour_lambda002_temp002/  # 48h results
     window_size_comparison_overall.png
     window_size_comparison_all_scenarios.png
```

---

## Troubleshooting

**Issue:** Out of memory during training  
**Fix:** Reduce `--batch-size` (try 16 instead of 32)

**Issue:** Different results than reported  
**Check:** 
- Data version (CERT r4.2)
- Python/PyTorch versions
- All hyperparameters match exactly

**Issue:** Training takes much longer  
**Possible:** CPU vs GPU (experiments run on CPU ~10 min each)

**Issue:** Import errors  
**Fix:** Ensure you're in the repo root and have installed dependencies

---

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Status:**  Validated and reproducible  
**Best Configuration:** 24-hour windows (exp012_lambda002_temp002)  
**Performance:** PR-AUC=0.714, Precision=1.00 (zero false alarms), Recall=0.56
