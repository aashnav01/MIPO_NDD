# Claude handoff: research and complete NDD assay metadata

## Your task

Research the experimental context of the assays in the attached `assay_metadata.csv`, fill values supported by primary papers and experiment records, and return usable updated files. Do the research and edits, not just propose a plan. Prioritize neurodevelopmental-disorder (NDD) assays, then support assays. Minimize unnecessary questions. Never replace an unknown with a plausible guess simply to make the table look complete.

The user is building a PreMode-inspired multi-task protein-variant model called MIPO-NDD, trained on Google Colab. Metadata quality matters because an assay-conditioned prediction head consumes **host, selection, system, treatment, and region**. Missing context is currently encoded as an unknown token.

This is experimental metadata curation, not clinical advice or pathogenicity assignment. A higher assay score is not necessarily healthier or more normal.

## Files and current state

Workspace: `C:\Users\shaba\Downloads\ndd_training_corpus_v3_package`

Use **`v4/data/assay_metadata.csv`**, not the historical 31-assay template at `prepared/audit/assay_metadata.csv`.

The current V4 metadata has **625 assays**:

| Source | Assays | Current context |
|---|---:|---|
| MaveDB Domainome | 501 | Shared study protocol already populated; verify rather than replacing indiscriminately |
| ProteinGym v1.3 | 99 | Partial reference-table curation; primary-paper research is the main opportunity |
| Original V3 collapsed consensus | 25 | Constituent assays and aggregation provenance absent from supplied files |

Current unknown counts across the five model fields: host **111**, selection **26**, system **84**, treatment **120**, region **25**. These are overlapping field counts, not numbers of distinct incomplete assays.

Baseline metadata SHA-256: `bb98120b8b25e78d8b83f5f94515668d7c23f5cb3f9233e3e9f3566793784095`.

The earlier reference-table enrichment is applied. **The additional literature findings below have NOT yet been applied to the metadata.** Some primary full texts and bibliographic records have been downloaded locally for research. Do not claim all papers or all assay-condition mappings have been reviewed.

The full corpus has 994,673 measurements, 495 genes, and 598 exact protein references. Its verified NDD view has 318,274 measurements across 135 genes. Preserve all measurements and their scores during this metadata-only task.

Useful files:

- `v4/data/assay_metadata.csv`: authoritative input/output table.
- `v4/sources/proteingym_reference.csv`: source assay IDs, DOI, raw phenotype column, raw directionality, mutated region, reference sequence, selection descriptions.
- `v4/data/protein_references.csv`: exact constructs and sequence mappings.
- `v4/data/gene_evidence.csv`: NDD inclusion evidence.
- `v4/sources/mavedb/`: score-set JSON records and score tables.
- `v4/sources/literature/index.json`: DOI-to-PMCID lookup and full-text availability.
- `v4/sources/literature/PMC*.txt` and `.xml`: available primary-paper text, for local inspection.
- `README_ndd_training_corpus_v3.txt`: explicit warning that consensus rows are collapsed targets, not original assay measurements.
- `mipo/metadata_curation.py`: current conservative automatic enrichment.
- `scripts/curate_v4_metadata.py`: metadata-only update and coverage report.
- `scripts/build_v4_dataset.py`: dataset rebuild also calls metadata enrichment.
- `scripts/research_assay_sources.py`: reproducible Europe PMC discovery and download helper.
- `v4/DATA_DICTIONARY.md`, `v4/DATASET_CARD.md`: schema and interpretation.

The companion `claude_metadata_research_bundle.zip` contains the metadata, reference tables, code, and this handoff. It omits the large measurement corpus and full paper text; use the supplied DOI/PMCID index to retrieve papers if you do not have the local workspace.

## Research procedure

1. Read the current table and group by study/DOI. Review the shared protocol once, then apply it only to the assay IDs it actually supports.
2. Start with `source == ProteinGym_v1.3` and `is_direct_ndd == 1`. Then cover the remaining ProteinGym support assays. Check the 25 consensus rows for traceable original provenance, without borrowing another assay's context.
3. Match each row using **assay ID, exact construct, raw phenotype column, and experimental arm**. A paper may contain multiple cell lines, drugs, doses, expression backgrounds, and follow-up assays.
4. Read the Methods, figure legends, supplement, and original score-table header. Use MaveDB, GEO, author repositories, and deposited protocols where available. Prefer primary sources over review articles and search snippets.
5. For every changed field, record its old value, new value, evidence URL/DOI, exact section/table locator, a short paraphrased explanation, and whether the evidence is assay-specific or only study-level. Add an access date.
6. Use a curated override file keyed by **`assay_key`** and an explicit allowed-field list. If grouping by DOI to prepare updates, resolve that group to exact assay keys before applying it. Never join on gene alone.
7. Leave unresolved fields as `unknown`, with a separate reason. `not_applicable` is valid only when the field genuinely does not apply. `none` or `untreated` requires an explicitly documented untreated arm. Do not use missingness labels as invented experimental conditions.

### Suggested field conventions

- **host:** organism hosting the phenotype assay; keep existing broad categories such as `yeast` where appropriate, and put the precise species/strain in additional columns. Never infer host from the species of the assayed protein. Bacterial cloning or lentiviral packaging cells are not automatically the assay host.
- **cell_line / strain:** optional new columns for exact biological context, e.g. K562 or PA-TU-8902. These are initially documentation columns, not automatically model inputs.
- **system:** assay technology, such as `DHFR_PCA`, `VAMP_seq`, `bacterial_two_hybrid`, or `cDNA_display_proteolysis`. Distinguish expression from actual phenotyping.
- **selection:** measured or selected phenotype. Preserve the source wording in `selection_assay` and record any standardized replacement separately or in the change log.
- **treatment:** the perturbation of the scored experimental arm. Keep dose, units, duration, and induction versus selection roles in separate columns when known. Avoid inserting every reagent mentioned in the paper.
- **region:** biological assay scope. Preserve exact reference coordinates separately. The existing length-based `domain`/`assay_reference` annotations are coarse and are not proof of an experimentally defined domain.
- **orientation / score_direction:** preserve source orientation unless you have independently established a correction. Any proposed sign change or target reinterpretation belongs in a separate issues report, not a silent metadata edit.

Add `field_status` or a separate per-field ledger distinguishing `verified_assay_specific`, `verified_shared_protocol`, `not_applicable`, `condition_mapping_unresolved`, `not_reported_in_reviewed_sources`, and `original_provenance_missing`. Merely failing to retrieve a paper does not establish that information was not reported.

## Findings already located: verify exact scope and apply

These notes are starting evidence, not permission to copy a whole paper's conditions into every associated assay. Original publication links and methods locators are included.

| Assay(s) | Finding to incorporate | Primary source / locator |
|---|---|---|
| All `*_Tsuboyama_2023_*` ProteinGym rows | Cell-free cDNA-display proteolysis; trypsin and chymotrypsin probe isolated-domain folding stability. Cellular host is not applicable; add `assay_environment=cell_free`. Do not assign E. coli merely because expression chemistry is bacterial-derived. | [Tsuboyama 2023](https://www.nature.com/articles/s41586-023-06328-6), Methods: proteolysis; PMCID PMC10412457 |
| `PPM1D_HUMAN_Miller_2022` | Engineered K562 cells with wild-type TP53 and a CDKN1A/p21-GFP reporter. PPM1D residues 1–427 were screened; 200 nM daunorubicin for 24 h before GFP-high/low sorting. The reporter's raw sign is not synonymous with phosphatase activity: inspect ProteinGym's transformation before interpreting direction. | [Miller 2022](https://www.nature.com/articles/s41467-022-30463-9), Results and Methods: saturation mutagenesis; PMC9246869 |
| `SHOC2_HUMAN_Kwon_2022` | The fitness screen is associated with PA-TU-8902 cells; MIA PaCa-2 is used in follow-up validation. The DMS protocol uses 10 nM trametinib versus DMSO, with harvest after 16 days. Verify the exact supplementary score-table mapping before setting the cell-line column. | [Kwon 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9694338/), SHOC2 DMS Viability Screen and Extended Data Fig. 9b |
| `RASH_HUMAN_Bandaru_2017` | Bacterial two-hybrid coupling of Ras/Raf binding to chloramphenicol resistance. The paper compares GAP/GEF regulatory settings: resolve the imported raw phenotype before labeling a particular setting. | [Bandaru 2017](https://elifesciences.org/articles/27810), Fig. 1 and bacterial two-hybrid Methods; PMC5538825 |
| `PTEN_HUMAN_Mighell_2018` | Humanized yeast growth rescue of PI3K-dependent toxicity, with galactose-inducible expression; lipid-phosphatase proxy. | [Mighell 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5986715/), Fig. 1 and Methods |
| `PTEN_HUMAN_Matreyek_2021` | HEK293T landing-pad cells; EGFP fusion abundance measured by VAMP-seq/FACS. Combined libraries mean not every protocol detail should be represented as a single uniform treatment. | [Matreyek 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8518224/), VAMP-seq results and methods; [GEO experiment record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4830181) |
| `CBS_HUMAN_Sun_2020` | Yeast `cys4` deletion complementation. Study includes vitamin B6 conditions of 0, 1 and 400 ng/mL. Do not select a concentration based on the generic raw column name `score`; trace the ProteinGym import first. | [Sun 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6993387/), High-throughput complementation screening |
| `MTHR_HUMAN_Weile_2021` | Yeast `met13`/`fol3` deletion assay, growth without methionine; multiple folinate concentrations and Ala222/Val222 backgrounds. Host and complementation system are recoverable; dose/background mapping still needs checking. | [Weile 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8322931/), experimental atlas design |
| `OTC_HUMAN_Lo_2023` | Yeast `arg3` deletion complementation measured in the absence of arginine. | [Lo 2023](https://pubmed.ncbi.nlm.nih.gov/37146589/), assay figure and Methods; PMCID PMC10183466 |
| `DLG4_HUMAN_Faure_2021`, `GRB2_HUMAN_Faure_2021` | Study measures abundance and binding with split-DHFR complementation. Paper year is 2022 despite the assay IDs. Resolve which assay arm supplies the generic `fitness` column; do not relabel the task solely from the paper title. | [Faure 2022](https://www.nature.com/articles/s41586-022-04586-4); author-deposited [bindingPCA plasmid](https://www.addgene.org/183619/) and [abundancePCA plasmid](https://www.addgene.org/183621/) |
| `RASK_HUMAN_Weng_2022_abundance`, `RASK_HUMAN_Weng_2022_binding-DARPin_K55` | Yeast AbundancePCA/BindingPCA. Published methods specify methotrexate competition. Imported preprint identifiers differ; establish linkage to the final publication and exact arm before overriding details. | [Weng final paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10866706/), DOI 10.1038/s41586-023-06954-0; Methods and [author manuscript](https://www.nature.com/articles/s41586-023-06954-0_reference.pdf) |
| `TPK1_HUMAN_Weile_2017`, `CALM1_HUMAN_Weile_2017`, `SUMO1_HUMAN_Weile_2017`, `UBC9_HUMAN_Weile_2017` | Yeast functional complementation. Temperature-sensitive selection is described, but verify each gene's protocol before propagating a common temperature. | [Weile 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5740498/), Materials and Methods |
| `MK01_HUMAN_Brenan_2016` | MAPK1 variant library in A375 cells under doxycycline-inducible expression. Imported raw phenotype is `DOX_Average`, with raw directionality -1; do not automatically assign one inhibitor condition from another part of the study. | [Brenan 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5120861/), library and phenotype screen |
| `CAR11_HUMAN_Meitlis_2020_gof`, `_lof` | Human TMD8 cells. The study compares 50 nM ibrutinib with a DMSO arm; map GOF and LOF score tables separately. | [Meitlis 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7820631/), saturation genome editing protocol |
| `P53_HUMAN_Giacomelli_2018_*` | A549 cells with wild-type or null TP53 background; nutlin-3 and etoposide are separate experimental arms. Preserve those distinctions explicitly. | [Giacomelli 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6168352/), pooled selection screens |
| `ADRB2_HUMAN_Jones_2020` | ADRB2-knockout HEK293T-derived landing-pad cells; cAMP-responsive barcoded transcriptional reporter read by RNA sequencing, stimulated with isoproterenol. Imported raw phenotype is `0.625`; verify its units. Do not mistake the luciferase validation assay for the multiplexed readout. | [Jones 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7707821/), Fig. 1 and reporter methods |
| `ACE2_HUMAN_Chan_2020` | Human Expi293F cells; FACS following binding of SARS-CoV-2 RBD-sfGFP to surface ACE2. | [Chan 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7574912/), library selection |
| `CASP3_HUMAN_Roychowdhury_2020`, `CASP7_HUMAN_Roychowdhury_2020` | Proteins expressed in E. coli, then assayed after lysis in droplets using a fluorogenic substrate. Describe expression host separately from the in-vitro droplet phenotype environment. | [Roychowdhury/Romero paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8748541/), microfluidic screening |
| `ERBB2_HUMAN_Elazar_2016`, `GLPA_HUMAN_Elazar_2016`, `LYAM1_HUMAN_Elazar_2016` | E. coli membrane-insertion assay using TOXCAT-beta-lactamase and ampicillin selection. | [Elazar 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4786438/), TbL assay |
| `AICDA_HUMAN_Gajula_2014_3cycles` | E. coli mutation/selection assay with rifampin resistance, Sat-Sel-Seq. Existing reference phrase `bulk RNA-sequencing` should be reviewed against the actual selection/readout, not trusted blindly. | [Gajula 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4150791/), Sat-Sel-Seq methodology |
| `KCNJ2_MOUSE_Coyote-Maestas_2022_function`, `_surface` | Mouse target protein is expressed in human HEK293-derived cells. Surface antibody staining and voltage-dye functional sorting are distinct assays. | [Coyote-Maestas 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9273215/), Fig. 1 and sorting methods |
| `OPSD_HUMAN_Wan_2019` | HEK293 cells; antibody-based rhodopsin surface-expression FACS. | [Wan 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC7027811/), assay design |
| `TADBP_HUMAN_Bolognesi_2019` | S. cerevisiae BY4741; galactose-induced TDP-43 expression and growth/toxicity selection. Check imported score direction before biological interpretation. | [Bolognesi 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6744496/), strain and selection methods |
| `VKOR1_HUMAN_Chiasson_2020_abundance`, `_activity` | HEK293T-derived cells; separate abundance and vitamin-K-dependent activity reporters. Warfarin is discussed biologically but must not be assigned as the screen treatment without arm-specific evidence. | [Chiasson 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7462613/), cell culture and assay methods |

Additional source leads in the cached literature: TPMT VAMP-seq (PMC5980760), PPARG THP-1/CD36 assays (PMC5131844), APP amyloid nucleation (PMC9674652), and MSH2 drug-selection assays (PMC7820803). These require exact phenotype/condition mapping before applying fields. PPARG includes an integrated score trained using variant classifications; report the implications separately rather than treating all scores as interchangeable raw measurements.

For GDI1, PSAT1, NPC1 and RAF1, use the DOI/source IDs in the input table to find the primary protocol. Do not confuse another paper on the same gene with the imported experiment. Some preprints have since acquired journal publications.

## The 25 collapsed-consensus rows

The original README states that `gate_consensus` scores are collapsed project targets and the supplied file no longer contains original per-assay measurements. Their protocol and biological orientation cannot be reconstructed just by finding a paper about the gene.

Look for original scripts, source manifests, raw tables, or a documented consensus mapping. If those are absent:

- Retain their measurement values and identifiers unchanged.
- Keep experimental fields unknown, with `original_provenance_missing` reasons.
- Preserve `orientation=as_supplied` and the uncertainty in biological direction.
- List the exact missing provenance required: constituent assay IDs, raw phenotype columns, score transforms/signs, aggregation weights/rules, and construct mapping.
- Record candidate source assays in a separate candidate table; never present them as confirmed assignments.

## Deliverables

1. `assay_metadata_researched.csv`: all 625 rows, original columns retained, added provenance/status columns allowed.
2. `assay_context_overrides.json`: exact-assay, field-level, source-backed updates suitable for deterministic replay.
3. `metadata_changes.csv`: one row per changed field (`assay_key,field,old_value,new_value,source_url,locator,evidence_summary,evidence_scope,review_date`).
4. `metadata_unresolved.csv`: one row per remaining unresolved field, what was searched, why unresolved, and the next source needed.
5. `METADATA_RESEARCH_REPORT.md`: before/after coverage, verified versus inferred/not-applicable counts, condition-mapping risks, and citations.

If only a chat upload is available, produce those files without claiming to have run the whole training package. If the repository is available, integrate the override replay with both `scripts/curate_v4_metadata.py` and `scripts/build_v4_dataset.py`, then verify rebuild compatibility.

**Replay warning:** the existing `enrich_metadata()` resets context provenance and re-extracts several fields from ProteinGym descriptions. Apply curated overrides AFTER automatic enrichment. A one-off CSV edit can otherwise be overwritten on rebuild. Add tests for exact-assay matching, missing-value handling, idempotence, and preservation of score semantics.

## Validation before returning

- Exactly 625 unique `assay_key` rows; no dropped, duplicated, or renamed assays.
- Preserve `gene`, `assay_id`, `protein_reference_id`, task, source provenance, tier, NDD flags, and `loaded_rows` unless a separate explicitly documented correction is necessary.
- No changes to measurement tables, scores, coordinates, orientation, or raw directionality as part of ordinary context completion.
- Every changed field has evidence; source-wide rules do not leak into unrelated assays or legacy consensus.
- Report unresolved fields honestly. Do not inflate completion by renaming every unknown to `none` or `not_applicable`.
- Run `python -m pytest -q` when the full repository is available. Previously, 20 tests passed; do not claim that result for new edits without rerunning.
- Metadata changes require a fresh training output directory, not resuming an old checkpoint with different inputs. Frozen ESM feature caches can be reused.
- Do not redistribute cached full-text papers in the training ZIP without checking their license. The handoff bundle deliberately excludes full texts.

Return the files and a short account of what was recovered. A useful partial, source-backed curation is preferable to a falsely complete table.
