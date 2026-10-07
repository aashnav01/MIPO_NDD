# MIPO-NDD: Colab research training package

**DGX A100 / multi-GPU Linux:** see [DGX_A100.md](DGX_A100.md). `scripts/dgx_v4_validation.py`
runs the same multi-gene validation protocol as the Colab launcher, with parallel GPU runs,
site-level feature-cache checks and resumable command-line launches.

**Multi-gene validation runner:** [MIPO_NDD_V4_Validation_Colab.ipynb](notebooks/MIPO_NDD_V4_Validation_Colab.ipynb)
uses five validation groups, three calibration groups, matched ablations, and
exact-row cached LLR comparisons. Upload `dist/mipo_ndd_v4_colab_v2.zip` directly
to Colab's `/content/` through the Files sidebar; the launcher defaults to local storage.
The Colab cell runs the same launcher as the DGX version.
See [TRAINING_UPDATE.md](TRAINING_UPDATE.md) for the protocol and task-coverage limits.

**V4 update:** use [MIPO_NDD_V4_Colab.ipynb](notebooks/MIPO_NDD_V4_Colab.ipynb) and [the V4 dataset card](v4/DATASET_CARD.md) for the expanded tiered dataset: 994,673 measurements, 495 genes, including a 135-gene evidence-filtered NDD view. All original V3 measurements are preserved. The updated pipeline supports separate construct references within one gene holdout. The remaining sections below document the original V3 release and its original coverage.

Start with **[notebooks/MIPO_NDD_Colab.ipynb](notebooks/MIPO_NDD_Colab.ipynb)**. The notebook installs the package, audits the corpus, uses the bundled validated sequences/structures, caches frozen ESM features, trains a gene-held-out model, evaluates it, and exports predictions and residue fields.

The model is **mutation impulse → residue perturbation field → latent bottleneck → metadata-conditioned assay readout**. It includes local geometric messages, long-range latent-landmark propagation, uncertainty, ranking supervision and optional paired-ensemble physical targets. The default network has **1,046,698 trainable parameters** with 1,280-dimensional frozen ESM2 features and unknown-only auxiliary metadata. Architecture decisions, equations, research positioning and limitations are in **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## What is actually in your data

| Measurement | Rows | Genes |
|---|---:|---:|
| Collapsed consensus | 136,734 | 25 |
| Fitness | 16,587 | 4 |
| Activity | 3,134 | 1 (HRAS) |
| Stability | 1,094 | 1 (UBR5) |
| Abundance / binding | 0 | 0 |

There are **157,549 rows, 31 genes and no gene with multiple independently retained measurement types**. Tier D teachers and generic pretraining examples are plans, not loaded training rows. The architecture supports additional measurements, but this corpus cannot by itself validate disentangled activity/abundance/binding mechanisms. Unsupported outputs are flagged.

All 31 reference sequences were checked against every observed WT site. KRAS uses the matching 188-residue ProteinGym sequence rather than the incompatible canonical reference. UBR5 is a 58-residue construct. Bundled structure arrays cover **30/31 genes**: 29 exact AlphaFold references and an exact 58-residue crop from experimental PDB 1I2T for UBR5. BRCA2 takes the explicitly masked sequence-only path. See `prepared/resources/structures/report.json` and the source records. Matching observed sites does not recover the lost original consensus-assay provenance.

## Files

- `mipo/`: validation, resource resolution, ESM extraction, datasets, model, losses, training, evaluation and inference.
- `configs/colab.json`: 256-node, 650M-feature default; `full_protein.json`: all residues; `curriculum.example.json`: optional broader-data stages.
- `prepared/`: audited coverage, sequences, structures, split definitions and existing-LLR baseline results.
- `scripts/run_experiments.py`: resumable folds, seeds and ablations.
- `scripts/ensemble.py`: combine seed predictions and recalibrate on separate calibration genes.
- `scripts/prepare_field_teachers.py`: optional physical-proxy labels from real paired WT/mutant structures.
- `tests/`: scientific invariants and complete synthetic training checks.
- `DATA_AND_EXPERIMENTS.md`: data expansion, homolog holdout, reporting and field-validation protocol.

## Local or Colab commands

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m mipo audit --corpus ndd_training_corpus_v3.csv --out artifacts/audit
python -m mipo features --corpus ndd_training_corpus_v3.csv --resources prepared/resources --cache artifacts/esm650
python -m mipo train --corpus ndd_training_corpus_v3.csv --resources prepared/resources --cache artifacts/esm650 --splits prepared/splits/gene.json --config configs/colab.json --out artifacts/fold0 --fold 0 --metadata prepared/audit/assay_metadata.csv
```

Append `--resume` to restart at the last completed epoch. Inputs, feature contents and configuration must match. An interrupted feature job resumes completed protein embeddings and batches of 50 masked sites. Training writes `last.pt`, `best.pt`, environment/provenance records, validation history, calibration predictions and untouched outer-test predictions. Keep feature caches and run outputs on Google Drive across Colab disconnections.

For a faster first GPU experiment, use `facebook/esm2_t30_150M_UR50D`, `--no-llr`, and a copied config with both `use_llr=false` and `require_llr=false`. Use a separate cache/output path for the 650M experiment. Existing supplied LLR values are not silently mixed with newly computed LLRs: their exact scoring provenance is unknown.

```bash
python scripts/run_experiments.py --corpus ndd_training_corpus_v3.csv --resources prepared/resources --cache artifacts/esm650 --splits prepared/splits/gene.json --config configs/colab.json --metadata prepared/audit/assay_metadata.csv --out artifacts/experiments --folds 0 --seeds 42 --ablations esm_mlp,mipo
```

Change `--folds all --seeds 42,123,2026` for the complete benchmark. This is a substantial experiment, not a single short Colab session. First validate one fold and measure runtime. Each fold holds out one test gene, one validation gene and one calibration gene; all other genes train. Fixed presets avoid test-based tuning. New extensive hyperparameter searches require further nested validation.

## Inference

Input CSV columns: `gene,wt,ref_pos,mut,task`, optionally `assay_key`. Positions refer to the validated experimental sequence and start at 1. Tasks: `consensus,activity,stability,abundance,binding,fitness`.

```bash
python -m mipo predict --variants queries.csv --resources prepared/resources --cache artifacts/esm650 --checkpoint artifacts/fold0/best.pt --out artifacts/predictions --export-fields
```

For new positions/proteins, build their validated sequence/feature cache first. The feature command accepts the normal corpus schema; `score_value` is ignored by feature extraction. Use an explicitly marked dummy value only in a separate inference-only file, never in training data. Task support flags must accompany predictions. Sigma is predictive score dispersion, not a clinical probability or a confidence interval by itself. Residue fields are hypotheses, not experimentally established allosteric pathways.

## Validation status

The package has been exercised locally on CPU with synthetic fixtures; the provided real-data LLR baseline was also computed. **The ESM650 model has not been trained or benchmarked on a Colab GPU here.** No accuracy improvement or publication novelty is claimed. `VALIDATION.md` records completed checks and remaining empirical work. The original corpus files are preserved.
