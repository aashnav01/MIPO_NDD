NDD TRAINING CORPUS V3
======================

Physical variant rows: 157,549
Loaded genes: 31
A_NDD_GOLD: 136,734 rows
B_DIRECT_NDD_FUNCTIONAL: 19,721 rows
C_DIRECT_NDD_STABILITY: 1,094 rows

IMPORTANT SCORE SEMANTICS
- score_value is always populated.
- For Tier A gate_target rows, score_value is the project's collapsed consensus y, not an original assay-level DMS score.
- For ProteinGym expansion rows, score_value/raw_dms_score is the assay-level ProteinGym DMS_score.
- train_target_0_1 is the common training target already present in v2.
- Do not interpret blank raw_dms_score on Tier A as missing experimental data: the source file supplied here no longer contains the original per-assay measurements.
- ESM LLR fields are complete for Tier A and pending for the ProteinGym expansion rows.

CORPUS LAYERS
A_NDD_GOLD: original 25-gene audited consensus target.
B_DIRECT_NDD_FUNCTIONAL: direct NDD functional ProteinGym additions.
C_DIRECT_NDD_STABILITY: direct NDD stability assays, kept separate.
D_NDD_FAMILY_TEACHER: identified teacher assays; variant rows listed in manifest until fetched.
Generic human DMS pretraining remains a separate pretraining layer and must not be mislabeled as NDD supervision.
