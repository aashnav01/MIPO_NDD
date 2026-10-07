# Recovering provenance for the collapsed-consensus targets

**Date:** 2026-09-21
**Scope:** the 25 `gate_consensus` assays, 136,734 measurements, previously carrying no
experimental context and `score_direction = as_supplied_unknown_biological_direction`.

## The idea

The V3 corpus supplied these targets rescaled to 0–1 with their provenance stripped. A
**rank-preserving rescale destroys values but not order.** So a candidate source can be tested
directly: if a deposited score set covers the same variants and its scores are a monotone
function of the supplied targets, it *is* the source.

The test is Spearman ρ on overlapping variants, searched over reference-numbering offsets.
ρ = ±1.0 at high coverage is proof of a rank-preserving relabel. Implemented in
`scripts/recover_consensus_provenance.py`; read-only with respect to the corpus.

## Result

**13 of 25 consensus assays identified — 53,207 of 136,734 rows (38.9%).**

| Gene | Rows | ρ | Coverage | Recovered assay |
|---|---:|---:|---:|---|
| CRX | 5,284 | **+1.0000** | 100% | HEK293-derived cells, synthetic fluorescent reporter |
| ASPA | 5,843 | **+1.0000** | 100% | VAMP-seq abundance, HEK293T landing pad |
| MAPK1 | 6,810 | **−1.0000** | 100% | A375, doxycycline-induced proliferation screen |
| SPOP | 6,261 | **−1.0000** | 100% | yeast survival assay |
| LARGE1 | 4,484 | +1.0000 | 100% | SMuRF, α-dystroglycan glycosylation |
| FKRP | 2,878 | +1.0000 | 100% | SMuRF, α-dystroglycan glycosylation |
| OTC | 1,570 | +1.0000 | 100% | yeast *arg3Δ* solid-growth complementation |
| SUMO1 | 1,919 | +0.9999 | 100% | yeast complementation, DMS-TileSeq |
| BRCA2 | 462 | +1.0000 | 100% | HDR assay in **V-C8 (Chinese hamster)** cells |
| KCNQ1 | 62 | +1.0000 | 100% | normalised currents, electrophysiology |
| PALB2 | 6,718 | +0.9965 | 100% | **mouse embryonic stem cells**, PARPi sensitivity |
| PSAT1 | 1,915 | +0.9935 | 99.9% | yeast solid-growth complementation |
| TSC2 | 9,001 | +0.9920 | 98.8% | Tuberin + RapGAP domain combined scores |

Ten match at |ρ| = 1.0 with complete coverage — these are single assays relabelled, not
aggregates. Three match at ρ ≈ 0.99 (`strong_partial`): near-certain but not exact, consistent
with light aggregation or filtering differences. They are recorded at
`condition_mapping_unresolved` status rather than `verified`.

**Still unresolved (12 genes, 83,527 rows):** ADSL, AIRE, BRCA1, CBS, HMGCR, JAG1, KCNE1,
KCNQ2, KRAS, MTHFR, PAX6, PTEN. AIRE and JAG1 have no matching target gene in MaveDB at all.
The other ten have best-candidate ρ between 0.78 and 0.97 at high coverage — **which is itself
evidence that those really are aggregates**, as the name implied. CBS (ρ=0.957 against a
single B6 arm) and MTHFR (ρ=0.888 against one folate/background arm) are the clearest cases:
the study deposited several arms and the consensus appears to average them.

## Two findings that matter for training

### 1. Two genes are rank-inverted relative to their source

`MAPK1` and `SPOP` match at **ρ = −1.0**. The supplied target is ordered *opposite* to the
deposited scores.

The corpus is internally consistent — consensus and the ProteinGym copy of MAPK1 agree at
ρ = +1.0, because ProteinGym also records `raw_DMS_directionality = -1`. So this is **not a
live training bug.** It is a trap for anyone who later joins raw MaveDB scores onto these rows,
which would silently mix orientations. Recorded as
`recovered_orientation = inverted_relative_to_source`; a test asserts a negative ρ is always
declared. No measurement was changed.

### 2. Three consensus assays duplicate data already in the corpus

`OTC`, `PSAT1` and `MAPK1` consensus rows are the same measurements already present as
`OTC_HUMAN_Lo_2023`, `SERC_HUMAN_Xie_2023` and `MK01_HUMAN_Brenan_2016` — confirmed at
ρ = +1.0, +0.9935 and +1.0 against their in-corpus counterparts.

The build script already caught this: all three are flagged
`possible_consensus_constituent_overlap = 1` and down-weighted to 0.25. The recovery confirms
the flag was correct and upgrades it from suspicion to certainty.

**These pairs are scientifically useful, not just redundant.** The same measurement appears
twice under two different monotone transforms — raw scale and 0–1 squash — with identical
ranks. That is a ready-made test of whether a metadata-conditioned observation model handles
assay scale, using data you already have.

## What changed in the metadata

| Field | Before | After first curation | After recovery |
|---|---:|---:|---:|
| host | 111 unknown | 25 | **16** |
| selection | 26 | 25 | **13** |
| system | 84 | 25 | **13** |
| treatment | 120 | 33 | **27** |
| region | 25 | 25 | **13** |

Unresolved model fields: the supplied research queue listed **366**; **82** remain.

Host composition also shifted: human cell lines rose from 18.8% to **20.6%** of rows, and two
hosts appeared that were previously invisible — **mouse ES cells** (PALB2) and **Chinese
hamster V-C8** (BRCA2). Both are cases where a human protein is assayed in a non-human host,
exactly the distinction the curation guards against.

## Correction to the earlier report

`METADATA_RESEARCH_REPORT.md` stated that consensus provenance "cannot be recovered by
research". **That was too strong.** What was established is that the *supplied files* contain
no provenance. It was not tested whether the constituent assays were independently
identifiable. For 13 of 25 they are. The report has been corrected.

## Reproduce

```bash
python scripts/recover_consensus_provenance.py   # rank-match against MaveDB (network, ~10 min)
python scripts/build_assay_overrides.py          # fold results into the override document
python scripts/curate_v4_metadata.py             # replay onto the metadata table
python -m pytest -q                              # 31 passed
```

Verified: reset to the pristine baseline `bb98120b…`, rebuilt end-to-end, **byte-identical
output**. No measurement, score, coordinate, orientation or raw directionality was modified;
recovery is recorded in added columns only.

## What would close the remaining 83,527 rows

- **CBS, MTHFR, ADSL, PAX6, KCNE1, KCNQ2** — the deposits contain multiple arms. Test whether
  a *mean of ranks* across arms reproduces the consensus, rather than any single arm.
- **BRCA1, KRAS, PTEN** — best candidates cover only 32–68% of variants; likely aggregates of
  several score sets. Test combinations.
- **AIRE, JAG1** — absent from MaveDB. Would need the original publications or author deposits.
