# Multi-gene validation update

Open `notebooks/MIPO_NDD_V4_Validation_Colab.ipynb` in a GPU Colab runtime. Use its
Files sidebar to upload `dist/mipo_ndd_v4_colab_v2.zip` to `/content/`, then run
the main cell. No Google Drive mount is required. Alternatively, paste all of
`scripts/colab_v4_validation.py` into one cell. The cell installs the release and runs
`scripts/dgx_v4_validation.py`, the same launcher used on a DGX (see `DGX_A100.md`), so
both environments share the feature-cache checks, matched comparisons and resume logic.

The release ZIP contains code, data and reference resources, not the precomputed
ESM embeddings/LLR cache from your previous Drive run. Optionally upload that
cache as `v4_650M_masked.zip` too (manifest.json at the ZIP root or inside
v4_650M_masked/). It is extracted, checked protein by protein and site by site, and
completed where needed. Without it, the cell builds features before training (for all
V4 proteins by default, so a later curriculum run never changes files that earlier runs
fingerprinted; `FEATURE_SCOPE='scope'` builds only the current corpus). Set
`DO_FEATURES=True` to always run the resumable extraction; `False` requires a complete
cache. Do not change model size or disable LLR merely to bypass this preparation if
comparing against the previous run. Set `PLAN_ONLY=True` to check the plan without GPU work.

Outputs are in `/content/v4_validation_...`. Run the notebook's download cell to save
results and checkpoints to your computer. Runtime storage is temporary. A second
download cell saves the feature cache for a future session. For Drive storage, set
`STORAGE='drive'` and place the release ZIP (and any feature ZIP) under
`MyDrive/MIPO_NDD/`; features, checkpoints and results then survive disconnects, and
training reads a local copy of the verified cache because Drive is slow for random reads.

Defaults preserve the requested training budget (3000 batches of 8 per epoch,
20 maximum direct-training epochs, patience 6) and architecture. Main MIPO and
three ablations run once each. Increase `SEEDS` and `TEST_GENES` only as part of a
predefined evaluation plan; runtime scales with models, seeds and folds.

## Split and checkpoint rules

- Five validation groups and three calibration groups by default. Existing
  validation/calibration groups stay in their roles; outer tests stay unchanged.
- Expansion uses task presence, favors scarce tasks when coverage counts tie,
  then uses seeded tie-breaking and the original eligible
  evaluation gene set. It preserves gene/cluster boundaries and retains at least
  one training group for each currently supported training task. It fails if
  the requested expansion cannot meet those constraints. It does not use target
  values or test performance. Task coverage is preferred, not guaranteed.
- The selection score is the assay-average Spearman within each validation gene,
  then the equal average across genes. Every validation gene must have a defined
  score. Calibration and test scores never select the checkpoint.
- Expanded holdouts reduce and change training data. Compare ablations within
  the new protocol; changes from the old PTEN score cannot be attributed solely
  to better checkpoint selection. The shipped clusters are not a complete
  homology-family holdout benchmark.
- Runs already early-stopped stay stopped on resume. Code, splits, configuration,
  metadata and feature provenance protect against mixing incompatible runs.
  Old checkpoints need the old release; use a new output folder for this update.

## New diagnostics

- `split_composition.csv`: role/gene/assay/task row counts.
- `task_support.json`: number of genes per task in each role. The Colab cell
  displays this before training. For the default direct PTEN fold, the only
  remaining abundance gene stays in training, so abundance has no separate
  validation or calibration support. Its calibrated intervals remain absent.
- `target_transform.json`: training-only centers and scales. Held-out target
  statistics never fit the transform.
- `history.json`: aggregate loss and its NLL, Huber, ranking, field and auxiliary
  components; per-gene/per-task validation scores; sample draws; stopping state.
- `validation_epochs/epoch_*.csv`: per-assay validation diagnostics each epoch.
  `validation_metrics.csv` and `validation_summary.json` describe the selected checkpoint.
- `training_status.json`: actual best epoch and early-stopping/epoch-cap reason.
- Test/calibration metrics include bias, target/prediction dispersion, predicted
  sigma and Gaussian 90% coverage. Test metrics additionally report calibrated
  interval coverage, width and supported row counts; absent calibration is missing,
  never substituted with Gaussian intervals.
- `matched_llr_metrics.csv`: trained and signed cached-LLR correlations on the
  exact same variants and targets. Partial legacy LLR columns cannot reduce the
  headline comparison to consensus. Undefined correlations suppress the headline.

Absolute metrics use training-task-transformed scores. Predictions also include
raw-score means, sigma and calibrated bounds. This adds visibility into scale
transfer; it does not establish or repair calibration on unseen genes. Empirical
calibration intervals have no assured coverage under gene distribution shift.

The entry point supports multiple seeds/folds, reports each seed separately, and
does not label a plateau as proof of memorization or automatically demand longer
training. Repeated seeds are not independent new test genes.

The companion notebook is generated from the Python cell with
`python scripts/build_validation_notebook.py`. Package the updated release with
`python scripts/package_v4_release.py --out dist/mipo_ndd_v4_colab_upload.zip`.
