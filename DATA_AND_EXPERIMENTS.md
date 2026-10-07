# Data contracts and experiment protocol

**Version note:** this document describes the V3 input and expansion protocol. The completed V4 data, evidence rules and updated workflows are documented in [v4/DATASET_CARD.md](v4/DATASET_CARD.md). Use the V4 notebook for the current tiered release.

## Existing corpus

The original CSV and manifest are unchanged. Read `prepared/audit/audit.json` for source SHA-256, row counts, target coverage and missing LLR counts. `prepared/audit/coverage.csv` is the gene/assay inventory. `gate_consensus` is scoped by gene, avoiding accidental across-gene ranking pairs. Tier A's blank raw score is legitimate: the preserved label is the collapsed consensus value.

`assay_metadata.csv` contains only known task/score semantics and explicit `unknown` context. Fill host, selection, system, treatment and region from the actual study methods. Use controlled vocabulary such as `yeast`, `growth`, `pca`, `none`, `domain`; avoid gene names or study-specific labels disguised as categories. `orientation=as_supplied` records that biological sign has not been reinterpreted. It is descriptive and does not automatically flip training labels.

Score ranges and scale differences are not normalized using the test set. Before publication, recover the consensus-generation protocol, measurement orientation, original assay overlap and normalization provenance. Historical consensus construction could itself have used information beyond the current split; this package cannot reverse undocumented preprocessing.

## Sequence and structure references

Reference JSON contract:

```json
{"GENE": {"sequence": "ACDE...", "key": "sequence_sha256_prefix", "source": "reference provenance", "accession": "UniProt accession", "length": 100}}
```

Never fill unobserved reference residues by assembling WT letters from a variant table. The resolver uses experimental ProteinGym references for expansion assays and reviewed UniProt candidates for consensus genes, checking every observed WT position. An unresolved or ambiguous reference is an error. KRAS's explicit override is provided in `sequence_overrides.json` and documented in its resolved record.

The implementation currently requires one exact reference sequence per gene. If multiple assays use distinct isoforms/constructs for one gene, extend the key to `protein_reference_id` while keeping the original gene as the holdout group, or harmonize assay positions with an audited mapping before import. Do not split those constructs into independent holdout genes.

Structure cache: `resources/structures/<sequence_key>.npz` with `coords[C,L,3]` in angstroms, `confidence[C,L]` in [0,1], and a scalar `sequence`. AlphaFold pLDDT is scaled by 100. Experimental B factors must not be interpreted as pLDDT; import with `--confidence none`. PDB imports normally require a complete exact-sequence chain in every model.

For a known affinity tag, `--allow-subsequence` permits a **unique exact contiguous** match and records the crop policy. It does not perform approximate alignment. This was used for UBR5: PDB 1I2T includes an extra terminal `AHG`, while the experimental target is its first 58 residues. BRCA2 has no bundled exact full-reference structure; its geometry is masked. No artificial straight-chain coordinates or AlphaFold fragment stitching are used.

Multi-model PDB imports supply conformers. For BioEmu or MD trajectory output, export sequence-matched backbone models to a multi-model PDB using your trajectory tool, preserving units and chain identity. BioEmu is a separate optional environment; it is not a lightweight prerequisite for the core notebook. See the [official repository](https://github.com/microsoft/bioemu) for current generation commands, model limits and GPU requirements. Single AlphaFold structures are not conformational ensembles.

## Adding real human DMS and family teachers

Download the processed substitution assay CSVs from the [official ProteinGym distribution](https://github.com/OATML-Markslab/ProteinGym). Supply their directory and the matching release reference CSV. Example selection file:

```csv
DMS_id,gene,supervision_tier,assay_type,protein_group
OFFICIAL_ASSAY_ID,GENE,E_GENERIC_HUMAN_DMS,activity,enzyme
ANOTHER_OFFICIAL_ID,GENE2,D_NDD_FAMILY_TEACHER,organismal_fitness,channel
```

Those example IDs are schema placeholders, not existing downloaded assays. Select actual IDs from the reference. Verify organism, functional relevance, evidence tier, target sequence, duplicates and overlap with existing consensus sources.

```bash
python -m mipo import-proteingym --corpus ndd_training_corpus_v3.csv --reference prepared/resources/proteingym_reference.csv --assay-dir /path/to/processed_csvs --selection selected_assays.csv --output expanded_corpus.csv
python -m mipo audit --corpus expanded_corpus.csv --out artifacts/expanded_audit
python -m mipo resolve-sequences --corpus expanded_corpus.csv --out artifacts/expanded_resources --overrides sequence_overrides.json
python -m mipo splits --corpus expanded_corpus.csv --out artifacts/expanded_splits.json
```

The importer retains only canonical single substitutions, explicitly counts excluded multi-mutants, rejects duplicate variant-assay measurements, and preserves raw DMS scores. The expanded file is separate. Multiple measurements of the same variant in distinct assays remain separate rows sharing one protein representation.

Only after loading real generic/family data should you use `configs/curriculum.example.json`. All stages exclude the outer test, validation and calibration groups. Reusing a model pretrained on all corpus genes would invalidate the holdout; the runner trains each fold from fresh initialization. Test sequences can be encoded using frozen external models, but test experimental labels cannot supervise any phase.

## Homology holdouts

The ordinary split file is whole-gene LOGO, not a homology claim. An optional Colab/Linux workflow is:

```bash
mmseqs easy-cluster prepared/resources/proteins.fasta cluster_result tmp_mmseqs --min-seq-id 0.3 -c 0.8 --cov-mode 0 --cluster-mode 1
python scripts/mmseqs_clusters.py --tsv cluster_result_cluster.tsv --out clusters.csv
python -m mipo splits --corpus ndd_training_corpus_v3.csv --clusters clusters.csv --out artifacts/cluster_splits.json
```

These are starting thresholds; high coverage can miss domain homologs in multidomain proteins. Review domain families and cross-split sequence similarities, especially the short UBR5 construct and channel paralogs. Union manually curated family exclusions with sequence clusters where appropriate. Keep at least four groups. Identical exact sequences across roles are rejected during training even in gene mode.

## Ablation sequence

1. Audit and signed supplied-LLR baseline. On the provided consensus genes, the computed macro-gene Spearman is about **0.4118 for 650M LLR** and **0.3709 for 150M LLR**. This is an existing-feature baseline, not evidence that the new model works; expansion-gene LLRs are missing.
2. Generate a consistent feature/LLR cache, then rerun `mipo baselines --resources ... --cache ...` for fair cached zero-shot comparisons.
3. Train ESM MLP and full MIPO on identical fixed folds, initially one seed and one fold to measure memory/runtime.
4. Run all 31 outer folds and three seeds, followed by chemistry, fixed-head, structure, global-kernel and LLR ablations. Compare consensus genes separately from singleton measurement types.
5. Repeat with family/sequence-cluster holdout and expanded real assays. Only then test ensemble and physical-teacher variants.

The same source assay appearing indirectly in a collapsed consensus and directly in an expansion must be tracked. Gene holdout contains same-gene leakage, but it does not make two correlated training labels independent evidence. Report this overlap and avoid overstating supervision volume.

Aggregate one architecture at a time:

```bash
python -m mipo aggregate --runs artifacts/experiments/mipo/seed_42/fold_00 artifacts/experiments/mipo/seed_42/fold_01 --out artifacts/report_mipo
python scripts/ensemble.py --runs artifacts/experiments/mipo/seed_42/fold_00 artifacts/experiments/mipo/seed_123/fold_00 artifacts/experiments/mipo/seed_2026/fold_00 --out artifacts/ensemble_fold00
```

Do not mix different architectures into a single gene-bootstrap summary. Averaging repeated seeds per gene is permitted; comparing architectures should use paired gene-level differences and the same defined-gene set. Missing/constant correlations need counts, not silent exclusion. Add independent external datasets before claiming clinically useful prediction.

## Physical field teachers

Create a CSV of `gene,wt,ref_pos,mut,wt_pdb,mut_pdb`, optionally `chain`. WT and mutant sequences must match their respective structures exactly. Run:

```bash
python scripts/prepare_field_teachers.py --manifest paired_structures.csv --resources prepared/resources --out artifacts/field_teachers
```

Set `auxiliary_dir` and `auxiliary_weight` in a separate experiment config. Only training-gene variants read these labels. Contact-degree change can use a single structure pair; RMSF change requires ensembles. Generated conformers carry model bias; label their source, model version and sampling settings. Do not evaluate physical-channel accuracy on the same structures used to create training targets.

## Limits that code cannot remove

- More protein/assay supervision, especially multiple properties on the same proteins, is required for mechanism disentanglement.
- Test-gene calibration is unknown without held-out-gene calibration evidence; variant counts do not provide independent gene sample size.
- Metadata cannot recover an unidentified assay's absolute measurement scale.
- Residue response magnitude is a learned signal. Without independent localization experiments, it is not proof of a pathway.
- Foundation-model training overlap and original corpus curation biases need a separate benchmark audit.
