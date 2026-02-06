# Test coverage map (tests only; main code unchanged)

This file lists what is tested and what remains. Add new test cases here as edge cases are discovered. Use `pytest tests/ -v` and `pytest tests/ --cov=mv_ueba` (with pytest-cov) to check coverage.

## Audit result (current suite)

- **test_basic.py**: All 5 tests salvageable; kept as-is.
- **test_manifold.py**: One fix applied: `normal_deviation(z)` expects a single point `(d,)`; test now loops over points. All 2 tests pass.
- **test_scoring.py**: Both tests salvageable; kept.
- **test_trajectory.py**: One test; salvageable; kept.

All 10 tests pass after the single manifold test fix.

---

## Covered (current tests)

| Module            | Function / class                     | Test file      | Notes                          |
|-------------------|--------------------------------------|----------------|--------------------------------|
| `mv_ueba`         | `__version__`                        | test_basic     | Present, non-empty string      |
| cnn_model         | `UEBACNNAutoencoder.forward`         | test_basic     | 2-channel I/O, latent shape    |
| cnn_model         | `sequence_mse_2d`                    | test_basic     | Shape (B,)                      |
| cnn_model         | `compute_stats` (data.py)            | test_basic     | mu, sigma (F,)                 |
| data              | `SeqDataset`                         | test_basic     | __len__, __getitem__ shape      |
| data              | `MaskValueSeqDataset`                | test_basic     | 2-channel, mask/value          |
| scoring           | `score_windows_mask_value`           | test_scoring   | Keys, shapes, finite            |
| latent_manifold   | `UEBALatentManifold` + `normal_deviation` | test_manifold | Single point (d,)              |
| latent_manifold   | `UEBALatentManifold.density_score`    | test_manifold  | Scalar, finite, >= 0            |
| trajectory        | `TrajectoryAnalyzer.score_trajectory`| test_trajectory| Smoke with manifold             |

---

## Not covered (candidates for new tests)

| Module     | Function / class                          | Priority | Suggested test type        |
|------------|-------------------------------------------|----------|----------------------------|
| cnn_model  | `two_term_mse_loss`                       | high     | Shape, finite, zero mask   |
| cnn_model  | `masked_value_mse`                        | high     | Shape, active-cell only    |
| cnn_model  | `MaskValueLoss`                           | high     | Forward, keys               |
| cnn_model  | `weighted_mse_loss`                       | medium   | Shape, finite               |
| cnn_model  | `prepare_ueba_sequences_for_cnn`           | medium   | (B,T,F) -> (B,2,T,F)       |
| cnn_model  | `create_ueba_cnn_model`                   | low      | Smoke, config               |
| data       | `WeightedSeqDataset`                      | medium   | __getitem__ (z, raw)       |
| data       | `TemporalPairedMaskValueSeqDataset`      | medium   | Pairs, shapes              |
| scoring    | `score_windows_standard`                 | medium   | Keys, shapes                |
| trajectory | `TrajectoryConfig`                        | low      | Defaults                   |
| trajectory | `create_trajectories_from_sequences`     | medium   | Grouping by user, length    |
| latent_manifold | `UEBAManifoldConfig`                 | low      | Defaults                    |
| latent_manifold | `get_manifold_info`                   | low      | Dict keys                   |
| etl        | `CERTFixedWindowLoader`                   | low      | Optional: integration only |

---

## How to add new test cases

1. **Parametrize for variants**  
   Use `@pytest.mark.parametrize("shape,edge", [...])` for different shapes or edge cases (e.g. single sample, zero activity).

2. **One assertion focus**  
   Prefer one logical behavior per test; add new test functions for new behaviors so the suite stays easy to extend.

3. **Docstrings**  
   Use NumPy-style docstrings for test functions (Summary, Parameters if any, Returns/assertions).

4. **No main code changes**  
   Only add or modify files under `tests/`.
