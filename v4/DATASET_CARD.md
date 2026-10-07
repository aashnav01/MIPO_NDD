# NDD training corpus V4

Built on 2026-09-19. This release expands the original corpus with **downloaded, validated measurements**, preserves score semantics, and separates NDD evidence from generic protein-effect supervision.

## What to use

- **Start in Colab:** `notebooks/MIPO_NDD_V4_Colab.ipynb`.
- **Verified NDD-only training:** `v4/data/ndd_direct_v4.csv.gz` — 318,274 measurements, 135 genes.
- **Full tiered training:** `v4/data/ndd_training_corpus_v4.csv.gz` — 994,673 measurements, 495 gene groups, 598 exact protein references and 625 assay groups.
- **Fast analytical loading:** the same master table is available as `ndd_training_corpus_v4.parquet`.
- **Separate tier files:** five CSV.gz files in `v4/data/`.
- **Interpretation and joins:** `assay_metadata.csv`, `protein_references.csv`, `gene_evidence.csv` and `multi_property_variant_index.csv.gz`.

Every original V3 measurement is retained, with `v3_row_id` linking back to its original row. The original source CSV is unchanged. Revised evidence labels and exact construct references are explicit additions, not claims that original assay-level labels have been recovered from the consensus itself.

## The five tiers

| Tier | Meaning | Measurements | Genes |
|---|---|---:|---:|
| A_LEGACY_CONSENSUS | Original 25-gene collapsed consensus, retained as auxiliary supervision | 136,734 | 25 |
| B_DIRECT_NDD_FUNCTIONAL | Assay-level functional, abundance, binding or fitness measurements on evidence-qualified human NDD genes | 74,255 | 15 |
| C_DIRECT_NDD_STABILITY | Stability measurements or explicitly labeled domain-abundance stability proxies on evidence-qualified human NDD genes | 159,715 | 116 |
| D_NDD_FAMILY_TEACHER | Explicitly selected related-protein/biological-neighborhood teachers; includes separately identified mouse ortholog experiments | 83,735 | 13 |
| E_GENERIC_HUMAN_DMS | Other human DMS/domain measurements for broad pretraining | 540,234 | 343 |

Gene counts overlap across tiers; do not add them to estimate unique genes. Tier names describe supervision roles, not degrees of clinical pathogenicity. Tier C's 116 genes are **not** 116 full-length functional maps.

The original name `A_NDD_GOLD` is retained in `legacy_supervision_tier`, but V4 uses `A_LEGACY_CONSENSUS`: a historical project designation is not independent NDD evidence. Eleven legacy genes do not satisfy this release's selected evidence rule: AIRE, BRCA1, BRCA2, CRX, HMGCR, JAG1, KCNE1, KCNQ1, PALB2, SUMO1 and TPK1. Their measurements remain available. This does **not** establish that those genes can never contribute to neurodevelopmental phenotypes.

## NDD evidence rule

A human gene qualifies for the NDD view if it meets either criterion:

1. Green/confidence 3 on **Genomics England PanelApp intellectual disability panel 285, version 11.31**, downloaded 2026-09-19.
2. A **DDG2P 2026-08-28** association with definitive, strong or moderate confidence and an explicit selected NDD phenotype: intellectual disability, developmental/speech delay, autism or neurodevelopmental-delay HPO terms, or an explicitly named neurodevelopmental/intellectual/developmental-delay/encephalopathy disorder.

The exact selected HPO IDs and text rule are encoded in `scripts/build_v4_dataset.py`. Evidence IDs, disease names, inheritance and mechanisms are retained in `gene_evidence.csv`. The union is a reproducible research inclusion rule, not an exhaustive or uniquely correct definition of NDD. Panel absence and missing HPO annotations can exclude relevant genes; review these cases rather than silently promoting them.

Panel membership is evidence about **gene–disorder association**. It does not imply that every missense variant causes disease, that every assay captures the disease mechanism, or that higher assay score means healthier function. Inheritance and gain-/loss-of-function mechanisms remain relevant to interpretation. PPM1D in particular retains the earlier project's warning that its assay may not represent the relevant NDD disease mechanism directly; use its experiment-specific score without treating it as a clinical effect label.

Primary evidence resources: [PanelApp panel 285](https://panelapp.genomicsengland.co.uk/panels/285/), [official PanelApp API](https://panelapp.genomicsengland.co.uk/api/v1/panels/285/), and [EBI DDG2P release directory](https://ftp.ebi.ac.uk/pub/databases/gene2phenotype/G2P_data_downloads/2026_08_28/).

## Actual source data added

**ProteinGym v1.3:** 96 human assays plus three explicitly selected mouse teacher assays. Only canonical single substitutions with finite scores are imported. Existing V3 assay rows are compared numerically and reused, not duplicated. The complete mutant sequence is checked against the supplied experimental WT sequence and substitution. Both raw source category and selection descriptions remain in the assay table. [Official ProteinGym distribution](https://github.com/OATML-Markslab/ProteinGym).

**Human Domainome 1.0 through MaveDB:** 521 candidate score sets were downloaded; 501 meet the one-human-protein/verified-UniProt-gene requirements. Twenty nonhuman/control or unmapped-target assays are excluded with reasons in `audit/excluded_assays.csv`. Published target sequences, normalized scores, source raw scores and error estimates are preserved. Nonsense, synonymous/control, unsupported HGVS and nonfinite-score rows are counted and excluded from the missense training table. [Domainome study](https://www.nature.com/articles/s41586-024-08370-4), [MaveDB API documentation](https://api.mavedb.org/docs).

Domainome uses growth coupled to isolated-domain abundance through a DHFR protein-complementation system. V4 represents this as `growth_coupled_domain_abundance_stability_proxy` within the stability task, with `host=yeast`, `system=DHFR_PCA`, `selection=growth`, `treatment=methotrexate` and `region=isolated_domain`. This is **not a thermodynamic ΔΔG measurement**, an independent full-length abundance measurement or a full-protein activity map.

MaveDB source score sets provide their individual licenses; the downloaded Domainome sources are CC0. ProteinGym and its underlying studies retain their own terms. The original user-provided corpus's licensing was not supplied. Source URLs and hashes are recorded; no blanket license is invented for the combined dataset.

## Measurement coverage

| Model task | Measurements |
|---|---:|
| Stability / stability proxy | 538,060 |
| Fitness | 140,624 |
| Consensus | 136,734 |
| Activity | 89,676 |
| Abundance | 64,356 |
| Binding | 25,223 |

There are **16 genes with more than one retained non-consensus measurement type** and **46,681 exact-reference variants measured in multiple task types**. These counts include teacher/generic data; they are not counts of NDD-only paired variants. See `audit/multi_property_genes.csv` for the gene-level support flags. Examples include PTEN activity/abundance and KRAS abundance/binding. An engineered binder assay must not automatically be interpreted as a native interaction assay; the assay description remains authoritative.

This is substantially better suited to multi-task learning than V3, but independent multi-property NDD overlap remains much smaller than the complete row count. Most new NDD genes enter through short-domain stability proxies. Report performance by source, scope and measurement type.

## Precise identity and numbering

`gene` is the biological holdout group. Human and selected mouse ortholog experiments use the same uppercase gene group, so an ortholog teacher cannot leak into training when that gene is held out.

`protein_reference_id = GENE__taxonomy__sequence_hash_prefix` identifies one exact experimental construct. The same gene can have full-length, domain or alternate-reference sequences. All are kept together during gene holdout, but have separate feature caches and position systems.

`ref_pos` is **1-based within that exact experimental reference**. It is never silently interpreted as a canonical full-length position. `canonical_pos` is populated only when the complete construct has one exact contiguous match in the retrieved UniProt canonical sequence. Otherwise it is blank. The source-reported offset is preserved separately in the reference table; an unverified offset is not promoted to canonical numbering.

Each row's WT letter matches its reference sequence. `variant_key` is construct-specific; `measurement_id` identifies its observation in a source assay. `assay_key` is gene plus assay ID. Measurements are never averaged merely because they share a variant or broad task.

Sequence references come from the experimental source. Available structures are matched or cropped only through an exact unique sequence match; unmatched references take the missing-structure path. Structure crops are reference-compatible approximations, not proof that an isolated domain adopts the same conformation in its cellular assay. `resources/structures/provenance.json` records structure sources and failures. There are 562 available structure references out of 598 in this build.

## Consensus overlap and weighting

The original collapsed consensus may have incorporated some of the newly downloaded assays, but its original constituent mapping is unavailable. V4 flags **60,535 consensus rows** whose exact-reference variant also has a retained assay-level measurement. Their default loss weight is **0.25**; other rows have weight 1. This weight is a transparent heuristic, not an estimated independence probability.

The pipeline applies this weight to regression, likelihood and within-assay ranking losses. Missing labels are never filled from another assay. Perform sensitivity analyses with those consensus rows excluded and with the entire consensus tier excluded; retain the master table for provenance. Same-gene holdouts prevent those rows from crossing the test boundary, but do not make overlapping training labels independent experiments.

## Training views and splits

The NDD-only view selects `is_direct_ndd=1`, including the verified subset of legacy consensus. The full view adds teacher/generic/legacy support rows. Separate fixed split manifests are supplied for both views; they differ in corpus fingerprint.

Outer test, validation and calibration groups are selected from verified NDD genes. All other genes remain eligible for training. Exact identical sequences across different genes are unioned into a shared group. Domain references and orthologs inherit the original gene group. These safeguards do not replace sequence/domain homology-cluster holdouts; use the cluster workflow for a stronger family-generalization claim.

`configs/v4_curriculum.json` trains:

1. Generic human data and unverified legacy consensus as auxiliary pretraining.
2. Explicit family/biological-neighborhood teachers.
3. Verified NDD tiers, with the pretrained backbone frozen for the final stage.

The same outer holdouts apply in **every stage**. A global checkpoint pretrained on all NDD genes would invalidate these splits. Gene evidence, source assay IDs and tier IDs are not neural-network features. Rich source descriptions are retained for audit but not blindly embedded; only the controlled semantic metadata fields enter the current model.

Default Colab profiles use frozen ESM150 and no LLR. Switch to ESM650/masked LLR for a separate experiment after timing the first fold. Existing V3 LLRs are preserved for provenance but not mixed with newly generated features. Cached global context is the complete experimental reference, sometimes a short domain, not necessarily a full-length human protein.

## Remaining worklist

The original NDD watchlist was searched through MaveDB. MECP2 and UBE3A domain measurements are now loaded. DDX3X has DNA-level saturation-genome-editing score sets, retained as candidates rather than automatically converted into amino-acid labels: splicing/context effects and multiple nucleotide substitutions per amino-acid change require an explicit mapping policy. Other zero-hit keyword searches are recorded as “not located in this search,” not proof that no experiments exist.

The release does not fabricate data for SCN2A, SCN8A, GRIN genes or other requested targets. `data/research_watchlist.csv` states loaded/candidate/search status. The provided discovery/build scripts make future expansion reproducible.

## Validation and reproducibility

Read `audit/validation.json` for checks against the real dataset and `audit/summary.json` for counts. Original variant identities and numeric scores are compared exactly, duplicate keys are rejected, every WT site is checked and every supplied split is checked for gene/reference overlap. Source inclusion and row exclusions are counted in `source_row_accounting.csv`. Synthetic model tests and notebook validation are separate from real-data QC; no trained V4 accuracy is claimed.

The source snapshots were downloaded on the build date. The source hashes make the build auditable even when online databases later change. To rebuild from cached/downloaded sources, run the `v4_fetch_*`, `v4_discover_mavedb`, `v4_download_assays`, `build_v4_dataset`, `v4_fetch_structures`, and `validate_v4_dataset` scripts in the documented sequence in `REBUILD.md`.
