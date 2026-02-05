# Data organization for replication

This document tells maintainers and replicators how to organize the dataset so the pipeline and scripts find it. The code was developed on **CERT Insider Threat Test Dataset r4.2**; the same layout can be used for other datasets with compatible schema.

---

## 1. CERT r4.2 (development dataset)

**Source:** Carnegie Mellon University, Kilthub  
**URL:** https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247  

Download the dataset and extract it. The pipeline expects a single **data directory** (e.g. `data/cert/r4.2`) with the following layout:

```
<data_dir>/
├── answers/
│   └── insiders.csv          # Required: user, scenario, start, end, dataset (e.g. "4.2")
├── logon.csv                 # Required: date, user, pc, activity
├── device.csv                # Required: date, user, activity
├── http.csv                  # Required: date, user, url
├── file.csv                  # Required: date, user
└── email.csv                 # Required: date, user, to
```

**Date format:** `%m/%d/%Y %H:%M:%S` (e.g. `01/15/2014 09:30:00`).

**How to point the code at it:** Pass the path to `<data_dir>` as `--data-dir` to the pipeline, e.g.:

```bash
python examples/exp005_fixed_window_pipeline.py \
  --experiment exp015_unified_traj \
  --data-dir data/cert/r4.2 \
  ...
```

If your extracted CERT r4.2 lives elsewhere (e.g. `/path/to/cert/r4.2`), use that path as `--data-dir`. The scripts `scripts/analyze_aggregation_scores.py` and `scripts/plot_trajectory_pr_curves.py` read `data_dir` from `runs/exp015_unified_traj/config.json` after the pipeline has been run once.

---

## 2. Recommended layout in this repo

From the **repository root**:

1. Create a directory for data (e.g. `data/`). Do **not** commit the data (it is large and not redistributable); keep `data/` in `.gitignore`.
2. Under `data/`, create a folder for the dataset (e.g. `data/cert/r4.2/`).
3. Extract or copy the CERT r4.2 contents so that the paths above exist under `data/cert/r4.2/`.

Example:

```
<repo_root>/
  data/                    # Ignored by git
    cert/
      r4.2/
        answers/
          insiders.csv
        logon.csv
        device.csv
        http.csv
        file.csv
        email.csv
  examples/
  scripts/
  manifold_ueba/
  ...
```

Then run with `--data-dir data/cert/r4.2` from the repo root.

---

## 3. Using another dataset

The ETL in `manifold_ueba/etl/cert_fixed_window.py` is built for CERT r4.2. To use another dataset:

- Provide the same directory layout and file names, or
- Adapt the loader to your schema: same **output** contract (train/test windows, labels, metadata with `user_id`, `window_start`, `scenario`), and the same 12 feature columns (see `CERT_FEATURES` in `cert_fixed_window.py`).

The model itself is dataset-agnostic; only the data loader is CERT-specific.
