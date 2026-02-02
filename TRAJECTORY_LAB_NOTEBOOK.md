# Trajectory Detection Lab Notebook

## Context and Motivation

**Starting Point:**
- Window-level detection: PR-AUC 0.714 (strong, stable)
- Trajectory-level detection: PR-AUC 0.208 (weak, bottleneck)
- Goal: Improve trajectory detection for multi-day campaign identification

**Hypothesis:**
Trajectory detection is weak not due to curvature/geometry, but due to flat geometry problems:
1. Trajectory construction (how we build sequences)
2. Labeling semantics (what we call "malicious")
3. Scoring functionals (how we aggregate signals)

**Strategy:**
Phase 0 - Fix flat geometry before exploring curvature

---

## Phase 0: Unified Trajectory Construction

### Experiment: exp015_unified_traj

**Date:** 2026-02-02

**What We Did:**

1. **Unified Trajectory Builder** (`build_trajectories_from_metadata`)
   - Single function for both train and test
   - Per-user chronological sequences (sorted by `window_start`)
   - Handles optional `scenario` field gracefully (test has it, train doesn't)
   - Returns structured dicts instead of tuples

2. **Fixed Training Reference Statistics**
   - **Before:** Sequential chunks ignoring user boundaries
     ```python
     for i in range(0, len(train_latents) - T + 1, stride):
         chunk = train_latents[i:i + T]  # WRONG: mixes users
     ```
   - **After:** Per-user chronological trajectories
     ```python
     train_trajectories = build_trajectories_from_metadata(
         train_latents, train_metadata, labels=None, ...
     )  # CORRECT: respects user boundaries and time order
     ```

3. **Dual Labeling Infrastructure**
   - **Any-overlap:** `label_any = 1` if ANY window in trajectory is malicious
   - **Majority-overlap:** `label_majority = 1` if >=50% of windows are malicious
   - Tracks `mal_frac` for each trajectory

4. **Enhanced Diagnostics**
   - Distribution of `mal_frac` among positives (mean, median, p90)
   - Reveals label dilution characteristics

**Why We Did It:**

1. **Training Reference Was Wrong:** Old approach created pseudo-trajectories by sliding over all training windows sequentially, mixing different users' behavior. This corrupted the "normal reference distribution" for geodesic deviation scoring.

2. **Label Semantics Were Unclear:** Single "any-overlap" label doesn't distinguish between:
   - "One weird day" (1/6 malicious windows)
   - "Sustained campaign" (6/6 malicious windows)

3. **Needed Better Diagnostics:** Without `mal_frac` distribution, we couldn't tell if low performance was due to:
   - Bad scoring functional
   - Label dilution
   - Data characteristics

### Results

**Trajectory Detection Performance:**

| Metric | exp012 (old baseline) | exp015 (unified) | Change |
|--------|----------------------|------------------|--------|
| Any-Overlap PR-AUC | 0.208 | 0.229 | **+10%** |
| Any-Overlap ROC-AUC | ~0.50 | 0.531 | +6% |
| Majority-Overlap PR-AUC | N/A | 0.162 | (new) |
| Majority-Overlap ROC-AUC | N/A | 0.556 | (new) |

**Window-Level Performance (unchanged):**
- PR-AUC: 0.698 (stable, as expected)

**Trajectory Construction Stats:**
```
Training: 1942 normal trajectories (per-user sequences)
Test: 501 total trajectories
  - Any-overlap positives: 65 (13.0%)
  - Majority-overlap positives: 33 (6.6%)
```

**Key Discovery - Label Dilution:**
```
mal_frac among any-overlap positives:
  mean   = 0.497
  median = 0.500
  p90    = 0.833
```

**What This Means:**
- Typical "malicious trajectory" is only **50% malicious** (3 out of 6 windows)
- Only **10% of positives** are sustained campaigns (>5/6 windows malicious)
- Most detections are "one or two bad days" mixed with normal behavior
- **Label dilution is the primary bottleneck**, not the scoring functional

### Interpretation

**Why +10% Improvement?**
1. Training reference statistics now reflect real per-user temporal behavior
2. Geodesic deviation baseline is now properly calibrated against correct normal distribution

**Why Majority-Overlap PR-AUC is Lower (0.162)?**
- Harder task: requires sustained malicious activity (>=50% windows)
- Fewer positives (33 vs 65)
- More meaningful for campaign detection use case
- ROC-AUC slightly better (0.556 vs 0.531) suggests better discrimination when it matters

**Why Performance is Still Weak?**
The `mal_frac` diagnostics reveal the core problem:
- Current scoring functional (geodesic deviation) treats entire 6-day trajectory as a single unit
- Works well for sustained anomalies (rare: 10% of cases)
- Fails for diluted labels (common: 90% of cases with mean mal_frac=0.5)

### Conclusions

1. **Baseline is Now Valid:** Training reference statistics are correct; results are reproducible and interpretable.

2. **Label Dilution is the Bottleneck:** Median mal_frac=0.5 means most "malicious trajectories" are half normal. Current geodesic deviation scoring cannot handle this.

3. **Dual Labeling Works:** Infrastructure is in place to evaluate both discovery (any-overlap) and campaign-focused (majority-overlap) use cases.

4. **Next Step is Clear:** Explore alternative scoring functionals that can handle diluted labels:
   - **Accumulation:** Aggregate per-window scores (sum, mean, top-k)
   - **Dynamics:** Trajectory velocity/acceleration norms
   - **Predictability:** Deviation from linear temporal model

---

## Phase 1: Flat Scoring Functionals (Planned)

### Goals
1. Test if better aggregation of window-level scores improves trajectory detection
2. Compare accumulation strategies (sum, mean, max, top-k)
3. Evaluate dynamics features (velocity, acceleration)
4. Try predictability deviation (linear model residuals)

### Hypothesis
Window-level scores are strong (PR-AUC 0.698). If we aggregate them intelligently along trajectories, we should capture "one or two bad days" better than single geodesic deviation score.

### Success Criteria
- Any-overlap PR-AUC > 0.30 (vs current 0.229)
- Identify which functional works best for diluted labels

---

## Experimental Log

| Exp ID | Description | Traj PR-AUC (Any) | Traj PR-AUC (Maj) | Window PR-AUC | Notes |
|--------|-------------|-------------------|-------------------|---------------|-------|
| exp012_lambda002_temp002 | Old baseline (broken train ref) | 0.208 | N/A | 0.714 | Sequential chunks, no user boundaries |
| exp015_unified_traj | Unified construction baseline | 0.229 | 0.162 | 0.698 | Per-user chrono sequences, dual labels |

---

## Code Changes Log

### 2026-02-02: Unified Trajectory Construction

**Files Modified:**
- `examples/exp005_fixed_window_pipeline.py`

**Key Changes:**

1. **New Function:** `build_trajectories_from_metadata()`
   - Location: Line 893-963
   - Replaces: `create_sliding_trajectories()` (test only) + sequential loop (train)
   - Handles both train and test with single unified logic

2. **Training Trajectories:**
   ```python
   # OLD (WRONG):
   for i in range(0, len(train_latents) - T + 1, stride):
       chunk = train_latents[i:i + T]
   
   # NEW (CORRECT):
   train_trajectories = build_trajectories_from_metadata(
       train_latents, train_metadata, labels=None, ...
   )
   ```

3. **Dual Label Evaluation:**
   - Evaluate with `label_any`
   - Evaluate with `label_majority`
   - Save both to JSON with config metadata

4. **Enhanced Logging:**
   - Trajectory counts by label type
   - `mal_frac` distribution statistics
   - Sanity checks for label inflation

**Function Signature:**
```python
def evaluate_trajectory_detection(
    model, manifold, 
    train_data, train_metadata,  # Added train_metadata
    test_data, test_labels, test_metadata, 
    mu, sigma, args, exp_dir
)
```

---

## Data Characteristics

**Dataset:** CERT r4.2 (filtered: 1 day < attack < 1 week)
- 26 insider threat users
- Scenario breakdown: 16 scenario_1, 10 scenario_3

**Window-Level:**
- Training: 5,929 normal 24hr windows
- Test: 1,609 windows (112 malicious = 7.0%)

**Trajectory-Level (T=6, stride=3):**
- Training: 1,942 normal trajectories
- Test: 501 trajectories
  - Any-overlap: 65 positive (13.0%)
  - Majority-overlap: 33 positive (6.6%)

**Label Characteristics:**
- Window-level imbalance: 7.0% malicious
- Trajectory any-overlap imbalance: 13.0% (inflated due to overlap)
- Trajectory majority-overlap imbalance: 6.6% (similar to window-level)

**mal_frac Distribution (any-overlap positives):**
- min: >0 (at least 1 malicious window required)
- median: 0.500 (typical positive is 50/50)
- mean: 0.497 (slightly below median, some diluted outliers)
- p90: 0.833 (only 10% are sustained campaigns)

This distribution explains why trajectory detection is hard: most positives look mostly normal.

---

## Next Session TODO

1. Implement accumulation scoring functionals (sum, mean, top-k of window scores)
2. Run ablation: compare accumulation strategies
3. Implement dynamics features (velocity, acceleration norms)
4. Compare all functionals on both label semantics (any vs majority)
5. Identify best performer for Phase 1 conclusion
