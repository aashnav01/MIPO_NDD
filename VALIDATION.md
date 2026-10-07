# Validation and delivery record

## V4 expanded release

Validated 994,673 measurements across 495 genes and 598 sequence references, including 318,274 verified NDD measurements across 135 genes. All 157,549 original rows and their scores are preserved exactly. No duplicate measurements, WT/reference mismatches, or nonhuman rows labeled direct NDD were found. Both 135-fold split sets passed gene and exact-sequence isolation checks. Structures are bundled for 562 references; 36 have explicit missing geometry. The updated test suite passes 19 tests. The V4 notebook passes schema and Python-cell compilation checks. V4 GPU training and scientific performance remain unexecuted.

The following sections document the original V3 implementation and its checks.

Date: 2026-09-19. Local environment: Windows, Python 3.10.11, PyTorch 2.11.0 CPU. The Colab GPU training run is intentionally left for the user's GPU environment; no trained scientific MIPO checkpoint is presented as a result.

## Completed

- Original corpus audited: 157,549 rows, 31 genes, 31 gene-scoped assay groups, no duplicate variant/assay keys and canonical single substitutions only.
- Original CSV SHA-256: `9fc238ea9a499f78694f5334a32284a2ab89d1b009a122fcf3a425a70878df2b`.
- All 31 sequences matched every observed WT site; KRAS reference mismatch resolved using the matching ProteinGym construct.
- 29 exact AlphaFold coordinate caches downloaded and validated. UBR5 uses an exact reference-matched crop from PDB 1I2T; BRCA2 is explicitly missing structure.
- Existing ESM650/ESM150 LLR baselines evaluated against the supplied consensus labels. No missing expansion LLR was imputed as a biological measurement.
- Editable package installation and CLI executed locally.
- Complete synthetic two-epoch train → validation → best checkpoint → calibration → test pipeline executed on CPU.
- **16 pytest cases passed**, including three model modes, finite backward gradients, rotation/reflection-consistent scalar/vector outputs, translation-invariant graph construction, conformer-order handling, mixed-conformer padding, zero mutation field, missing geometry, whole-gene/cluster split disjointness, duplicate and WT mismatch rejection, long-sequence window coverage, training-only transforms, unseen task handling, auxiliary-label isolation, and epoch-boundary resume equality with an uninterrupted run.
- Colab notebook schema validated with nbformat; every Python cell compiled. Runtime-dependent Drive/GPU cells are not claimed to have executed locally.

## Code review

Reviewed the ingestion/reference boundary, sparse graph construction, feature cache, model/loss interfaces, split/normalization boundaries, checkpoint serialization, inference and experiment scripts using the code-review-expert checklist. The source directory was initially data-only, so review covered the new files directly rather than an existing Git diff.

Issues found and corrected during implementation:

1. KRAS canonical/reference mismatch: reject incorrect mappings and provide the experimentally matching override.
2. Potential assay collision: group by gene plus assay ID, because `gate_consensus` is reused across genes.
3. Conformer-padding dependence: use an explicit conformer mask when pooling vector magnitudes.
4. Conformer-order dependence in graph selection: union neighbors across selected frames and rank by ensemble-averaged distance.
5. Resume integrity: reject configuration/input/feature-content changes before overwriting run provenance; compare actual interrupted and uninterrupted training results.
6. Unsupported measurement types: route through a trained generic fallback and flag unsupported outputs instead of presenting random untrained heads as biological estimates.
7. Teacher leakage: physical labels load only in the training dataset, with source fingerprints.

No known unresolved critical correctness issue was found in the exercised paths. This is an internal engineering review, not an independent scientific replication or exhaustive audit.

## Remaining empirical validation

- ESM650 extraction and training throughput, peak GPU memory and numerical behavior on the user's Colab hardware.
- Full real-data folds, multi-seed uncertainty, fixed baselines and homology-held-out comparisons.
- Independent field localization and physical-proxy validation.
- Rich assay metadata, genuine multi-property training overlap and additional human/teacher assays.
- External-model pretraining contamination, historical consensus provenance and a systematic novelty review.

The 8M ESM integration check, if completed, is recorded separately in `prepared/integration_check.json`; it does not substitute for the 650M Colab benchmark. No theoretical coverage guarantee under unseen-gene distribution shift is asserted.
