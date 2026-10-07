# MIPO-NDD V4 assay metadata: research and curation report

**Review date:** 2026-09-20
**Input:** `v4/data/assay_metadata.csv`, 625 assays, SHA-256 `bb98120b…784095` (matched the recorded baseline exactly)
**Output:** 625 assays, no rows added, dropped, renamed or reordered

This is experimental-context curation. Nothing here assigns pathogenicity, and a higher assay
score is not a healthier or more normal one. No measurement, score, coordinate, orientation or
raw directionality was changed.

---

## 1. Result

Counts are assays whose field is still `unknown`, before → after.

| Source | Assays | host | selection | system | treatment | region |
|---|---:|---|---|---|---|---|
| ProteinGym v1.3 | 99 | 86 → **0** | 1 → **0** | 59 → **0** | 95 → **8** | 0 → 0 |
| MaveDB Domainome | 501 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| Original V3 consensus | 25 | 25 → 25 | 25 → 25 | 25 → 25 | 25 → 25 | 25 → 25 |
| **All** | **625** | 111 → **25** | 26 → **25** | 84 → **25** | 120 → **33** | 25 → **25** |

**Every one of the 600 real-provenance assays now has host, selection, system and region.**
The only remaining gaps are 8 `treatment` fields (§5) and the 25 collapsed-consensus rows,
whose provenance was never supplied. **Correction (2026-09-21): 13 of those 25 have since been
recovered by rank matching against MaveDB — see `CONSENSUS_RECOVERY_REPORT.md`. The claim below
that consensus provenance cannot be recovered was too strong; what was established is that the
supplied files do not contain it.**

The supplied research queue (`v4/audit/metadata_research_queue.csv`) listed **366 open fields**.
133 remain, and 125 of those are the consensus rows — so **8 of the 241 researchable ProteinGym
fields are still open**.

`region` was never literally `unknown` outside the consensus rows, but it was **not meaningful**:
every ProteinGym row had been labelled `domain` or `assay_reference` purely by whether the
reference sequence was shorter than 100 residues. All 99 now carry a source-backed scope, and
all 600 non-consensus assays carry exact `region_coordinates`.

### The NDD priority set

| 21 direct-NDD ProteinGym assays | host | selection | system | treatment | region |
|---|---|---|---|---|---|
| before | 15 unknown | 0 | 14 unknown | 20 unknown | 0 |
| after | **0** | **0** | **0** | **2** | **0** |

### Evidence quality across all 3,125 model-field cells

| Status | Cells | Share |
|---|---:|---:|
| `verified_shared_protocol` | 2,653 | 84.9% |
| `verified_assay_specific` | 297 | 9.5% |
| `original_provenance_missing` (the 25 consensus rows) | 125 | 4.0% |
| `not_applicable` | 31 | 1.0% |
| `source_table_derived` | 11 | 0.4% |
| `not_reported_in_reviewed_sources` | 8 | 0.3% |

The shared-protocol share is dominated by the 501 Domainome assays, which legitimately share one
protocol. Nothing was marked verified without a source.

**On `not_applicable`:** 31 cells use it, every one backed by Methods text showing a single
population read on a reporter with no treated/untreated contrast. Where an assay applies a reagent
that *is* the measurement — a protease, a staining antibody, a fluorogenic substrate, an
immobilised ligand — that reagent is recorded as the probe in `system` or `construct_notes`, not
as a treatment. Anything less clear was left `unknown`. A test (`test_not_applicable_is_a_value_not_a_missingness_label`)
enforces that the label never stands in for a gap.

---

## 2. Deliverables

| File | Contents |
|---|---|
| `v4/curation/assay_metadata_researched.csv` | All 625 rows, original 38 columns kept, 18 provenance/status columns added |
| `v4/data/assay_context_overrides.json` | 600 assays, 4,902 field entries, each with value, status, URL, locator, summary and scope |
| `v4/curation/metadata_changes.csv` | 2,348 changed content fields, one row each, with evidence |
| `v4/curation/metadata_unresolved.csv` | 133 unresolved model fields, with what was searched and what is needed next |
| `v4/curation/metadata_issues.csv` | 21 direction / target-definition / condition-mapping concerns, reported not applied |
| `v4/curation/consensus_provenance_required.csv` | The exact provenance missing for each of the 25 consensus rows |
| `v4/curation/assay_metadata_baseline.csv` | Snapshot of the pre-curation table, so the change log stays reproducible |
| `v4/sources/mavedb_conditions/` | 32 cached MaveDB records, the primary evidence for arm identification |

Added columns: `host_status`, `selection_status`, `system_status`, `treatment_status`,
`region_status`, `assay_environment`, `expression_host`, `cell_line`, `strain`, `treatment_dose`,
`treatment_duration`, `treatment_role`, `expression_induction`, `region_coordinates`,
`construct_notes`, `curation_evidence`, `curation_review_date`, `curation_schema_version`.

The documentation columns (`cell_line`, `strain`, …) are deliberately **not** model inputs yet.

---

## 3. Method, and what actually unlocked the unknowns

Assays were grouped by study, the shared protocol read once, then resolved back to **exact
`assay_key` values** before anything was written — never a join on gene alone. Each row was
matched on assay ID, construct, raw phenotype column and experimental arm.

The decisive move was **MaveDB**, in two ways:

1. **ProteinGym sometimes names the score set it imported** in `raw_DMS_filename`
   (`urn_mavedb_00000049-a-6_scores.csv`). Eleven assays did. That URN pins dose, genetic
   background and readout in a way reading the paper cannot.
2. **For the rest, a gene search finds the same study's deposited score sets.** Twenty more assays
   were matched this way, each confirmed by publication DOI rather than by name similarity, and
   each recorded in `scripts/research_assay_conditions.py` with the reason it matches.

That single source resolved most of what the first sweep of the literature could not, because
deposited score sets describe *the arm*, while papers describe *the study*.

### Conditions recovered

| Assay | Recovered | How |
|---|---|---|
| `MTHR_HUMAN_Weile_2021` | 25 µg/mL folate, **WT (Ala222)** background | score set titled "MTHFR at 25ug/ml folate in WT background" |
| `ADRB2_HUMAN_Jones_2020` | The cryptic phenotype name `0.625` is **0.625 µM isoproterenol**, 4 h | paper dose range 0–10 µM |
| `RASH_HUMAN_Bandaru_2017` | Column `unregulated` = the arm with **no GAP and no GEF** | paper defines the term |
| `P53_HUMAN_Giacomelli_2018_*` | Per-arm doses differ: **2.5 µM**, **5 µM**, **5 µM** | Methods sample list |
| `SCN5A_HUMAN_Glazer_2019` | HEK TetBxb1BFP; **25 µM veratridine + 250 ng/mL brevetoxin + 10 µM ouabain, 5 h** | paper Methods |
| `CP2C9_HUMAN_Amorosi_2021` | abundance = VAMP-seq in **HEK293T**; activity = Click-seq in **humanized yeast** | two score sets |
| `HXK4_HUMAN_Gersing_*` | activity = complementation of an **hxk1Δhxk2Δglk1Δ** strain; abundance = **DHFR-PCA** | two score sets, two DOIs |
| `NPC1_HUMAN_Erwood_2022_*` | **saturation prime editing** in locally haploidized HEK293T / RPE1, **LysoTracker** gating | score sets |
| `MSH2_HUMAN_Jia_2020` | **HAP1** cells | score set titled "MSH2 LOF scores (HAP1)" |
| `NUD15_HUMAN_Suiter_2020` | HEK293T landing pad, **thioguanine** cytotoxicity arm (not the VAMP-seq arm) | score set + paper |
| `KCNE1_HUMAN_Muhammad_2023_function` | fitness in an **LP-KCNQ1-S140G** background — not conductance | score set methodText |
| `KCNH2_HUMAN_Kozek_2020` | HEK293T, anti-HA **surface trafficking** — not "Voltage" | paper Methods |
| `MET_HUMAN_Estevam_2023` | **Ba/F3 murine** cells, IL-3 withdrawal, exon-14-skipped TPR-MET | preprint Methods |
| `TPOR_HUMAN_Bridgford_2020` | **Ba/F3**, cytokine-independent growth, S505N background | paper |
| `S22A1_HUMAN_Yee_2023_activity` | **SM73 at 1 µM** — the `_1_` in `SM73_1_score` is the dose | preprint |
| `SC6A4_HUMAN_Young_2021` | Expi293F; `avg_MYC` is the **myc surface-expression** arm | preprint |
| `SRC_HUMAN_Nguyen_2022` | `diffsel` = **radicicol minus DMSO** differential | paper |
| `Elazar` ×3 | **ampicillin** selection (β-lactamase anchored in the inner membrane) | paper |
| `PPARG_HUMAN_Majithia_2016` | **THP-1** PPARG-null macrophages, **rosiglitazone / PGJ2** | paper |
| `CAR11_HUMAN_Meitlis_2020_lof` | the explicitly documented **DMSO vehicle** arm | paper |
| `OTC_HUMAN_Lo_2023` | arg3Δ0, solid SD **lacking arginine**, 30 °C, 72 h; leader aa 2–32 omitted | experiment record |

### Host values a careless pass would have got wrong

- **22 Tsuboyama assays** → `not_applicable` / `cell_free`. The translation kit is *E. coli*-derived, but no cell hosts the phenotype.
- **2 caspase assays** → expressed in *E. coli*, phenotyped in vitro in droplets after lysis.
- **`YAP1_Araya_2012`, `PAI1_Huttinger_2021`, `UBE4B_Starita_2013`** → phage display; selection happens on beads, *E. coli* only propagates phage.
- **`KCNJ2` ×2** → mouse protein, **human** HEK293T host.
- **`MET_Estevam_2023`, `TPOR_Bridgford_2020`** → human protein, **mouse** Ba/F3 host.
- **`CP2C9_activity`** → yeast, while the abundance arm of the *same study* is human HEK293T.

Final host distribution: yeast 528, human_cell_line 37, not_applicable 27, e_coli 6,
mouse_cell_line 2, unknown 25 (all consensus).

### Region, from a heuristic to coordinates

For the 501 Domainome assays, each MaveDB record names its source-study domain identifier
(`P00519_PF00018_64`); with the target-sequence length this yields an **exact Pfam domain span in
UniProt coordinates for all 501**. For ProteinGym assays, `region_mutated` gives the exact
mutagenized span — except where the paper contradicts it, as with BRCA1 (§4).

### Selection vocabulary

`selection` held 95 near-unique free-text strings, so semantically identical assays would have
become distinct tokens to the assay-conditioned head. All 99 ProteinGym assays now carry a
controlled value, and the verbatim source wording is untouched in `selection_assay`. Three
assays are noted in `selection_left_unstandardised` where the source names two properties at once.

---

## 4. Issues found — reported, not silently fixed

All 21 are in `metadata_issues.csv`. The ones that change how the data should be used:

- **`PPARG_HUMAN_Majithia_2016`** — `raw_DMS_filename` points at a MITER web query, and the score
  is an **integrated score built using variant classifications**. Treating it as a raw measurement
  risks circularity with the labels a model is meant to predict.
- **`HMDH_HUMAN_Jiang_2019`** — the imported set is titled *"rosuvastatin imputed and refined"*.
  Some values are imputed, not measured.
- **`PPM1D_HUMAN_Miller_2022`** — the score is log2(GFP-high/GFP-low), where **higher means
  impaired** phosphatase activity, while ProteinGym labels it `fitness` with directionality +1.
  Orientation left exactly as supplied.
- **`BRCA1_HUMAN_Findlay_2018`** — ProteinGym records `region_mutated` as 1–1855 of 1863, which a
  coverage rule reads as whole-protein. The editing actually targeted **thirteen exons encoding the
  RING and BRCT domains**. `region` was set from the paper, not the rule — and any assay whose
  region came only from `region_mutated` may carry the same over-statement.
- **`HXK4` (GCK) ×2** — ProteinGym lists the **same raw file and same `score` column** for both, but
  the deposited score sets prove they are different experiments from **different publications**.
- **`AICDA`** — the automatic enrichment matched "bulk RNA-sequencing" in ProteinGym's text and set
  `system = bulk_RNA_sequencing`. It is an *E. coli* rifampin selection read by DNA pyrosequencing.
  **Any other field set by that same phrase matcher deserves the same check.**
- **`KCNH2` / `KCNE1_function`** — ProteinGym's descriptions ("Voltage", "potassium channel
  function") do not match the deposited assays (surface trafficking; cell fitness).
- **`SRC_Nguyen_2022`** — `diffsel` is a radicicol-minus-DMSO **differential**, not a single-arm value.
- **`CD19_Klesmith_2019`** — the metadata DOI is the CD19-fusion retargeting paper, while the
  single-site FMC63 library is reported in the companion Biochemistry paper.
- **`DLG4` / `GRB2`** — imported as generic `OrganismalFitness`, but the study has distinct
  abundancePCA and bindingPCA arms. Task labels **left untouched**; the five context fields are
  common to both arms, so they were curated and the arm identity stays unresolved.

---

## 5. What is still unresolved, and why

**125 cells — the 25 collapsed-consensus rows.** All five fields stay `unknown` with status
`original_provenance_missing`. Measurement values, identifiers and `orientation = as_supplied` are
unchanged. `consensus_provenance_required.csv` lists, per row, the five things needed: constituent
assay identifiers, the raw phenotype column per constituent, the score transform and sign per
constituent, the aggregation weights or rules, and the construct/coordinate mapping. No candidate
source assays were asserted at the time of this report. **Superseded 2026-09-21:** a
rank-matching test (Spearman rho against deposited MaveDB score sets) identified the source
assay for 13 of the 25, covering 53,207 rows, two of them rank-inverted. See
`CONSENSUS_RECOVERY_REPORT.md`.

**8 cells — all `treatment`, all ProteinGym:**

| Assay | Why it is still open |
|---|---|
| `CBS_HUMAN_Sun_2020` | Study ran vitamin B6 at 0, 1 and 400 ng/mL; ProteinGym imported a column called only `score` with a blank `raw_DMS_filename`. Picking one would be a guess. |
| `HEM3_HUMAN_Loggerenberg_2023` | Two deposited isoform maps (ubiquitous / erythroid-specific); which one was imported is not established. |
| `CCR5_HUMAN_Gill_2023` | Two sort gates (myc surface expression, BiFC); `avg_score` does not say which. |
| `KCNE1_HUMAN_Muhammad_2023_function` | Fitness assay with no compound identified in the reviewed sources. |
| `RAF1_HUMAN_Zinkus-Boltz_2019` | PACS selection agents not established from the score-set record. |
| `CP2C9_HUMAN_Amorosi_2021_activity` | Click-seq probe not verified in an accessible source. |
| `B2L11_HUMAN_Dutta_2010`, `CD19_HUMAN_Klesmith_2019` | Yeast-display binding; the staining reagent is documented only at source-table level, and CD19 additionally has the citation mismatch above. |

**Failing to retrieve a paper does not mean the information was not reported.** Where a source was
simply unavailable the status is `not_reported_in_reviewed_sources` and the per-row
`next_source_needed` names the document required.

---

## 6. Replay, rebuild and verification

`enrich_metadata()` resets context provenance and re-derives fields from ProteinGym descriptions,
so a one-off CSV edit is destroyed on the next rebuild. Curated context is therefore applied
**after** it, in both entry points:

```python
# scripts/curate_v4_metadata.py and scripts/build_v4_dataset.py
assays = default_status(apply_overrides(enrich_metadata(assays)))
```

`apply_overrides()` matches on full `assay_key` only, validates every field against the override
file's `allowed_fields`, and refuses to touch a protected column (scores, identifiers, task,
orientation, `score_direction`, `loaded_rows`, tier, NDD flags, source provenance).

Verified on this run:

- 625 unique `assay_key` rows; none added, dropped, duplicated or renamed.
- 14 protected columns byte-identical to the baseline; asserted in the script itself.
- Reset to the pristine baseline and rebuilt end-to-end: **byte-identical output**. Running the
  curation twice produces the same SHA-256 — the pass is idempotent.
- `python -m pytest -q` → **30 passed** (20 before this work; 10 added, covering exact-assay
  matching, unknown-key handling, protected and non-allowed field rejection, idempotence across
  re-enrichment, gap labelling, override-file well-formedness, preservation of measurement
  semantics and consensus unknowns, and that `not_applicable` is never a missingness label).

Reproduce from the baseline with:

```bash
python scripts/research_assay_conditions.py   # cache MaveDB score-set/experiment records (network)
python scripts/build_assay_overrides.py       # author v4/data/assay_context_overrides.json
python scripts/curate_v4_metadata.py          # replay and write the deliverables
python -m pytest -q
```

**Training note:** these metadata changes alter model inputs, so they require a **fresh training
output directory** rather than resuming a checkpoint trained on the old context. Frozen ESM
feature caches are unaffected and can be reused.

**Licensing:** cached full-text papers under `v4/sources/literature/` were used only for local
reading and are not redistributed in any bundle. The cached MaveDB records under
`v4/sources/mavedb_conditions/` are metadata records from a public API.
