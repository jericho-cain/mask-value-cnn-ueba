# Window Size Ablation Study for UEBA Anomaly Detection

**Branch:** `feature/window-size-ablation`  
**Paper:** Temporal Window Size Selection for CNN-Based Behavioral Anomaly Detection

---

## Overview

This repository contains a complete, reproducible implementation of a window size ablation study investigating how temporal window length affects anomaly detection performance in User and Entity Behavior Analytics (UEBA).

**Research Question:** What is the optimal temporal window size for detecting insider threat attacks lasting 1-7 days?

**Method:** CNN autoencoder with manifold learning, tested on CERT r4.2 insider threat dataset

**Key Finding:** **24-hour windows achieve optimal performance** (PR-AUC 0.714), significantly outperforming both shorter (12h: 0.277) and longer (48h: 0.682) alternatives.

---

## Results Summary

| Window Size | Mask PR-AUC | Combined PR-AUC | ROC-AUC | Change vs 24h |
|-------------|-------------|-----------------|---------|---------------|
| **12-hour** | 0.277 | 0.282 | 0.736 | **-61%** ❌ |
| **24-hour** ⭐ | **0.714** | 0.716 | 0.849 | **baseline** ✅ |
| **48-hour** | 0.682 | 0.682 | 0.879 | **-4.5%** ⚠️ |

**Interpretation:**
- 12h windows: Insufficient context, too noisy
- 24h windows: **Optimal balance** between context and granularity
- 48h windows: Excessive aggregation loses temporal precision

### Scenario-Specific Performance

| Window | Scenario 1 (Logon+Device) | Scenario 3 (Logon+Removable) |
|--------|---------------------------|------------------------------|
| 12h | 0.178 | 0.377 |
| 24h | **0.674** | **0.697** |
| 48h | 0.677 | 0.331 ⚠️ |

**Notable:** 48h windows show **53% degradation** for removable media exfiltration (Scenario 3), indicating these attacks require finer temporal resolution.

---

## Quick Start

### Prerequisites

```bash
# Python 3.8+ with dependencies
pip install torch numpy pandas scikit-learn scipy matplotlib

# CERT r4.2 Dataset (download from CMU)
# https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
# Extract to: data/cert/r4.2/
```

### Run All Three Experiments

```bash
# 12-hour windows (~15 min)
python examples/exp005_fixed_window_pipeline.py \
    --experiment exp013_12hour_lambda002_temp002 \
    --data-dir data/cert/r4.2 --window-hours 12 \
    --epochs 50 --use-mask-value --lambda-value 0.02 \
    --use-temporal-reg --lambda-temporal 0.02 --grid-search

# 24-hour windows (~10 min) ⭐ OPTIMAL
python examples/exp005_fixed_window_pipeline.py \
    --experiment exp012_lambda002_temp002 \
    --data-dir data/cert/r4.2 --window-hours 24 \
    --epochs 50 --use-mask-value --lambda-value 0.02 \
    --use-temporal-reg --lambda-temporal 0.02 --grid-search

# 48-hour windows (~8 min)
python examples/exp005_fixed_window_pipeline.py \
    --experiment exp014_48hour_lambda002_temp002 \
    --data-dir data/cert/r4.2 --window-hours 48 \
    --epochs 50 --use-mask-value --lambda-value 0.02 \
    --use-temporal-reg --lambda-temporal 0.02 --grid-search
```

**Total runtime:** ~35 minutes (sequential)

---

## Methodology

### Architecture

**CNN Autoencoder:**
- **Input:** (T, F) where T = temporal buckets (12, 24, or 48), F = 12 behavioral features
- **Encoding:** Spatial compression via strided convolutions → latent_dim=32
- **Decoding:** Symmetric upsampling via transposed convolutions
- **Loss:** Dual-channel (mask + value) with temporal smoothness regularization

### Key Techniques

1. **Dual-Channel Loss:**
   - **Mask channel:** Binary presence/absence (BCEWithLogitsLoss)
   - **Value channel:** Z-scored magnitude (masked MSE)
   - **Weight:** λ_value = 0.02 (minimal value regularization)

2. **Temporal Regularization:**
   - Penalty for large jumps between consecutive windows: λ_temporal * ||z_t+1 - z_t||²
   - Encourages smooth latent dynamics for normal behavior
   - **Weight:** λ_temporal = 0.02

3. **Chronological Split:**
   - Training: Temporally earlier normal windows (80%)
   - Testing: Later normal + all malicious windows (20% + attacks)
   - Prevents temporal leakage

4. **Reproducibility:**
   - PyTorch seed = 42
   - NumPy seed = 42
   - Deterministic data split

---

## Repository Structure

```
manifold-ueba/
├── README.md                                    # This file
├── WINDOW_SIZE_ABLATION_README.md              # Detailed replication guide
├── examples/
│   └── exp005_fixed_window_pipeline.py         # Main experiment script
├── manifold_ueba/
│   ├── cnn_model.py                            # CNN autoencoder (flexible architecture)
│   ├── data.py                                 # Dataset classes with temporal pairing
│   ├── latent_manifold.py                      # Manifold construction
│   ├── trajectory.py                           # Geodesic deviation scoring
│   ├── scoring.py                              # Anomaly scoring utilities
│   └── etl/
│       └── cert_fixed_window.py                # CERT data loader (chronological split)
└── runs/
    ├── exp012_lambda002_temp002/               # 24h results ⭐
    ├── exp013_12hour_lambda002_temp002/        # 12h results
    ├── exp014_48hour_lambda002_temp002/        # 48h results
    ├── window_size_comparison_overall.png      # Comparison plot (3 curves)
    └── window_size_comparison_all_scenarios.png # Detailed comparison (6 curves)
```

---

## Visualizations

### Overall Comparison

![Window Size Comparison](runs/window_size_comparison_overall.png)

**Clean comparison showing 24-hour windows (green) significantly outperform 12-hour (red) and slightly outperform 48-hour (blue).**

### Scenario Breakdown

Individual PR curves by attack scenario available for each window size:
- [12-hour by scenario](runs/exp013_12hour_lambda002_temp002/pr_curve_by_scenario.png)
- [24-hour by scenario](runs/exp012_lambda002_temp002/pr_curve_by_scenario.png) ⭐
- [48-hour by scenario](runs/exp014_48hour_lambda002_temp002/pr_curve_by_scenario.png)

---

## Key Contributions

1. **Systematic window size ablation** on real-world insider threat data
2. **Flexible CNN architecture** supporting variable temporal resolutions
3. **Temporal consistency regularization** for improved latent representations
4. **Scenario-specific analysis** revealing differential window size effects
5. **Complete reproducibility** with deterministic splits and seeding

---

## Performance Details

### Window-Level Detection (Best: 24h)

**Experiment:** `exp012_lambda002_temp002`

| Metric | Value |
|--------|-------|
| **Mask PR-AUC** | **0.714** |
| **Combined PR-AUC** | 0.716 |
| **ROC-AUC** | 0.849 |
| **Precision** | 1.00 (zero false alarms) |
| **Recall** | 0.56 |
| **Best F1** | 0.72 |

**Operating point:** At optimal threshold, achieves **perfect precision** (no false positives) with 56% recall.

### Trajectory-Level Detection

| Window Size | Trajectory PR-AUC | Trajectory ROC-AUC |
|-------------|-------------------|--------------------|
| 12h | 0.071 | 0.385 |
| 24h | **0.208** | 0.533 |
| 48h | 0.197 | 0.384 |

**Note:** Trajectory detection (sequential geodesic deviation) shows improvement with 24h windows but remains limited by weak label dilution (6-day windows with "any malicious" labeling).

---

## Experimental Controls

All experiments use **identical hyperparameters** except window size:

- **Architecture:** CNN autoencoder, latent_dim=32
- **Loss weights:** λ_value=0.02, λ_temporal=0.02
- **Training:** 50 epochs, batch_size=32, lr=0.001
- **Data:** CERT r4.2, chronological 80/20 split, 7-day attack buffer
- **Evaluation:** Grid search over α/β combinations for optimal scoring

**Only variable:** `--window-hours` (12, 24, or 48)

---

## Dataset

**CERT Insider Threat Test Dataset r4.2**
- **Source:** Carnegie Mellon University
- **Size:** 32M+ events, 1000 users, 18 months
- **Attacks:** 70 compromised users, 26 with 1-7 day attack duration
- **Scenarios:** Logon+Device theft (16 users), Logon+Removable media (10 users)
- **Features:** 12 behavioral features (logon, device, HTTP, file, email)

---

## Documentation

- **[WINDOW_SIZE_ABLATION_README.md](WINDOW_SIZE_ABLATION_README.md)** - Complete replication instructions
- **Expected Outputs:** JSON results and PNG visualizations included in `runs/`
- **No trained models:** Excluded due to size; reproducible via training script

---

## Citation

If you use this work, please cite:

```bibtex
@article{cain2026window,
  title={Temporal Window Size Selection for CNN-Based Behavioral Anomaly Detection},
  author={Cain, Jericho},
  journal={[Journal TBD]},
  year={2026}
}
```

---

## Reproducibility Statement

This repository provides:
- ✅ Complete source code
- ✅ Exact hyperparameters
- ✅ Deterministic data splits (chronological, seeded)
- ✅ Pre-computed results (JSON + plots)
- ✅ Step-by-step replication instructions
- ✅ Expected outputs and runtimes

**Variation:** Results should be within ±2% PR-AUC due to hardware/library version differences. Data split and architecture are deterministic.

---

## Contact

For questions about this work, please open a GitHub issue or contact:
- Email: [your email]
- GitHub: [@jericho-cain](https://github.com/jericho-cain)

---

## License

MIT License - see LICENSE file for details.

---

**Last Updated:** January 2026  
**Status:** ✅ Paper submission ready  
**Branch:** `feature/window-size-ablation`
