# Handoff: mask-value-cnn-ueba (paper replication repo)

**For the next Cursor session / agent when working in the new repo.**

---

## What this repo is

- **mask-value-cnn-ueba**: Clean, paper-replication-only repo for the dual-channel (mask + value) CNN autoencoder and top-k aggregation work.
- Created from **manifold-ueba** by pushing the branch **feature/paper-replication** to this new repo. The original repo (manifold-ueba) is unchanged and still has all branches (main, feature/trajectory-detection, etc.).

## What’s in this repo

- **Entry points (README):**
  1. **Window ablation:** `examples/exp005_fixed_window_pipeline.py` with `--experiment exp012/exp013/exp014` and `--window-hours 24/12/48`.
  2. **Trajectory (exp015):** Same script with `--experiment exp015_unified_traj`, then `scripts/analyze_aggregation_scores.py`, then `scripts/plot_trajectory_pr_curves.py`.
- **Data:** CERT r4.2 (or compatible). Layout and `--data-dir` are documented in **DATA_ORGANIZATION.md**.
- **Manifold code:** Kept on purpose (UEBALatentManifold, TrajectoryAnalyzer, off-manifold β, geodesic baseline) for probing the latent space; top-k aggregation of mask_bce is the main result.
- **Package:** `manifold_ueba` (name unchanged for minimal code churn); API is mask-value CNN + data + scoring + latent_manifold + trajectory.

## Suggested next steps (in this repo)

1. **Optional:** Rename default branch to `main`:  
   `git branch -m feature/paper-replication main` then `git push origin main`, set default branch in GitHub to `main`, then delete `feature/paper-replication` if desired.
2. **Verify replication:** Follow README: install deps, prepare data per DATA_ORGANIZATION.md, run ablation (exp012/013/014) and/or exp015 + Phase 1 + PR curve scripts; confirm outputs and metrics match the paper.
3. **Tests:** Run `pytest tests/` (tests are test_basic, test_scoring, test_manifold, test_trajectory).

## Key files for replication

| Purpose | File |
|--------|------|
| Data layout | DATA_ORGANIZATION.md |
| All steps | README.md |
| Pipeline | examples/exp005_fixed_window_pipeline.py |
| Phase 1 aggregation | scripts/analyze_aggregation_scores.py |
| Trajectory PR curves | scripts/plot_trajectory_pr_curves.py |
| CERT loader | manifold_ueba/etl/cert_fixed_window.py |

## Don’t change (by design)

- No grid_search or manifold_scorer modules (removed as obsolete; pipeline has its own inline grid search).
- No trajectory_analysis.py (deprecated old script).
- Manifold and trajectory code stays; only obsolete helpers were removed.

---

*Last updated: when feature/paper-replication was pushed to mask-value-cnn-ueba.*
