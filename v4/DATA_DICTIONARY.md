# V4 schema

One row in the master table is one finite-score missense measurement in one assay. Empty CSV fields are missing data, not zero. Gzip CSV and Parquet contain the same measurements; Parquet preserves numeric target/position types while historical mixed-type source columns are represented as strings.

| Column | Meaning |
|---|---|
| measurement_id | Stable source + assay + source-row hash; observation key |
| gene | Uppercase gene holdout group; never a learned ID feature |
| protein_reference_id | Exact experimental sequence/construct identifier |
| wt, mut | Canonical one-letter amino acids |
| ref_pos | 1-based position in experimental reference |
| canonical_pos | 1-based UniProt canonical position only after unique exact sequence mapping; otherwise missing |
| position_system | Explicit `experimental_reference_1_based` |
| assay_id, assay_key | Source assay identifier and gene-scoped identifier |
| variant_key | Exact reference plus substitution; does not collapse constructs |
| task | consensus, activity, stability, abundance, binding or fitness |
| assay_type | Original-compatible task label used by the pipeline |
| measurement_type | Distinguishes physical stability, domain-abundance proxy, aggregation proxy and other measurement types |
| score_value | Preserved source target; no cross-assay averaging or new global normalization |
| score_kind | Collapsed consensus, ProteinGym DMS_score, or MaveDB Domainome normalized score |
| score_orientation | Experimental ordering; never automatically means healthy/pathogenic |
| measurement_sigma | Published error estimate in source score units when available; not a model prediction |
| source_raw_score, source_raw_sigma | Domainome pre-normalization values as provided by MaveDB |
| raw_dms_score | ProteinGym's processed DMS_score, preserved under the V3-compatible column name; not necessarily raw instrument data |
| consensus_y | Original collapsed consensus where provided |
| supervision_tier | A legacy consensus, B direct functional, C direct stability/proxy, D teacher, E generic |
| training_role | Explicit intended modeling role |
| is_direct_ndd | 1 only for a human source satisfying the stated evidence rule |
| ndd_evidence_class | Which evidence rule qualified the gene |
| taxon_id, species | Biological target species, not the assay host |
| source, source_row_id | Source collection and its zero-based row identifier |
| source_variant | Original ProteinGym mutant or MaveDB protein HGVS notation |
| v3_row_id | Original zero-based V3 row index; missing for newly imported observations |
| default_sample_weight | 0.25 for possible overlapping consensus observations, otherwise 1 |
| possible_consensus_constituent_overlap | Exact-reference variant overlap flag, not proof of historical assay membership |
| legacy_* | Preserved historical tier, NDD assertion or target normalization; not the current evidence decision |
| llr_650M, llr_150M | Original supplied LLRs where available; do not infer zeros for missing scores |
| assay_level_available | Whether the target is a retained assay-level observation rather than collapsed consensus |
| protein_group | Explicitly assigned teacher neighborhood where known; otherwise unassigned |
| n_sets, feature_status, provenance_note, dms_score_bin_source | Retained historical/source audit fields; not automatic model inputs |

## Normalized companion tables

`assay_metadata.csv` joins on `assay_key`. It stores source URLs/hashes, publication identifiers, experimental scope, host/selection/system/treatment, score direction, coarse source classifications, gene evidence and retained row counts. Unknown experimental host is not replaced by the target species.

`protein_references.csv` joins on `protein_reference_id`. It stores the full reference sequence once, sequence hash, taxonomy, accession, source-reported offset and independently checked canonical offset. `resources/sequences.json` is the model-ready equivalent.

`gene_evidence.csv` joins on `gene`. It records panel confidence/version, DDG2P qualifying associations, inheritance, disease mechanisms and evidence date. An `is_direct_ndd=0` row does not prove lack of biological relevance.

`multi_property_variant_index.csv.gz` joins on `variant_key`, lists tasks and assay counts, and contains no averaged target. Only non-consensus observations count toward its task multiplicity.

`research_watchlist.csv` records the original NDD target search results and why candidate datasets were loaded or deferred. It is separate from the training table: no placeholder measurement rows enter training.

Reported experimental sigma is preserved but the current training loss does not explicitly deconvolve it from model dispersion. No missing mechanistic target is inferred from it. Metadata fields not present in a source remain unknown.
# Assay-context curation update

Use `v4/data/assay_metadata.csv` with the V4 notebook. `prepared/audit/assay_metadata.csv` is the historical V3 template and is not the enriched V4 metadata.

ProteinGym `selection_assay` descriptions are now also populated in the model-consumed `selection` field. Explicit yeast mentions, selected experimental methods and named drugs populate `host`, `system` and `treatment`; unreported context stays `unknown`. Human target species never implies a human experimental host. Named treatments do not imply known doses. Method extraction is conservative and incomplete.

`context_status`, `context_limitations` and `context_evidence` record the level and source of curation. The 25 collapsed legacy consensus assays cannot be assigned a unique protocol or biological direction from the supplied information. `as_supplied` means no sign change and no verified biological interpretation; it does not mean higher is healthier. ProteinGym orientation follows its supplied DMS score, and Domainome orientation describes its published proxy. Scores were not changed by this metadata update.

`v4/audit/metadata_coverage.json` reports remaining unknowns. Changing training metadata requires a fresh training run/output directory rather than resuming a checkpoint with different inputs. Cached ESM features remain reusable.
