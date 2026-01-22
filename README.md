# Manifold UEBA

Geometric anomaly detection for User and Entity Behavior Analytics (UEBA) using CNN autoencoders and manifold learning.

## Overview

This repository implements a novel approach to behavioral anomaly detection that treats user activity sequences as points in a learned latent space, then uses differential geometry to detect anomalies that deviate from the "manifold of normal behavior."

**Key insight:** Traditional reconstruction-error-based approaches miss anomalies that are far from normal behavior but still easy to reconstruct. By combining reconstruction error (alpha) with off-manifold distance (beta), we capture both types of anomalies.

### Core Components

- **CNN Autoencoder** (`manifold_ueba/cnn_model.py`): Treats behavioral sequences as 2D "images" in (time, features) space
- **Latent Manifold** (`manifold_ueba/latent_manifold.py`): k-NN manifold with tangent space estimation for off-manifold distance
- **Manifold Scorer** (`manifold_ueba/manifold_scorer.py`): Combined scoring with tunable alpha/beta coefficients  
- **Trajectory Analyzer** (`manifold_ueba/trajectory.py`): Tracks behavioral drift over time via geodesic deviation

## Installation

```bash
# Clone repository
git clone https://github.com/jericho-cain/manifoldUEBA.git
cd manifold-ueba

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start with CERT Dataset

The repository includes a complete pipeline for the [CERT Insider Threat Dataset](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247).

### 1. Download CERT Data

Download from CMU (requires browser):
- `r4.2.tar.bz2` (main dataset)
- `answers.tar.bz2` (ground truth labels)

```bash
mkdir -p data/cert/r4.2
tar -xjf ~/Downloads/r4.2.tar.bz2 -C data/cert/r4.2 --strip-components=1
tar -xjf ~/Downloads/answers.tar.bz2 -C data/cert/r4.2
```

### 2. Process Data and Train Model

```bash
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
```

### 3. Run Trajectory Analysis

```bash
python examples/trajectory_analysis.py \
    --load-processed data/cert_full_1hr.npz \
    --load-model data/cert_model_1hr.pt \
    --load-manifold data/cert_manifold_1hr.npz \
    --min-attack-hours 24 \
    --window-size 6 \
    --stride 3
```

## Scoring Model

The anomaly score combines two components:

```
score = alpha * reconstruction_error + beta * off_manifold_distance
```

- **alpha (reconstruction):** How poorly the autoencoder reconstructs the sequence
- **beta (geometry):** How far the latent representation is from the manifold of normal behavior

Grid search over alpha/beta combinations finds optimal weights for your data.

## Project Structure

```
manifold-ueba/
├── manifold_ueba/
│   ├── __init__.py
│   ├── cnn_model.py          # CNN Autoencoder
│   ├── latent_manifold.py    # k-NN manifold, tangent spaces
│   ├── manifold_scorer.py    # Combined alpha/beta scoring
│   ├── trajectory.py         # Trajectory-based analysis
│   ├── grid_search.py        # Hyperparameter optimization
│   ├── data.py               # Data utilities
│   └── etl/
│       └── cert.py           # CERT dataset loader
├── examples/
│   ├── cert_data_pipeline.py     # Main experiment pipeline
│   └── trajectory_analysis.py    # Trajectory evaluation
├── docs/
│   ├── cert-experiment-notes.md
│   └── cnn-manifold-learning-implementation.md
├── tests/
└── requirements.txt
```

## Documentation

- [Experiment Notes](docs/cert-experiment-notes.md) - Methodology and findings
- [Implementation Details](docs/cnn-manifold-learning-implementation.md) - Technical architecture

## Citation

If you use this code in your research, please cite:

```bibtex
@software{manifold_ueba,
  author = {Cain, Jericho},
  title = {Manifold UEBA: Geometric Anomaly Detection for User Behavior Analytics},
  year = {2025},
  url = {https://github.com/jericho-cain/manifoldUEBA}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- CERT Insider Threat Dataset provided by Carnegie Mellon University
- Manifold learning approach inspired by techniques from gravitational wave astronomy
