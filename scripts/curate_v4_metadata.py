"""Update assay context without changing measurement files or split fingerprints.

Order matters: ``enrich_metadata`` re-derives a few fields from ProteinGym descriptions and
resets context provenance, so curated overrides are replayed *after* it. A one-off CSV edit
would be silently undone on the next rebuild; the override document is the durable record.

Writes the authoritative table plus the curation deliverables under v4/curation/.
"""
from pathlib import Path
import json

import pandas as pd

from mipo.common import save_json, digest
from mipo.metadata_curation import (MODEL_FIELDS, apply_overrides, default_status,
                                    enrich_metadata, load_overrides)

PATH = Path('v4/data/assay_metadata.csv')
CURATION = Path('v4/curation')
BASELINE_SHA = 'bb98120b8b25e78d8b83f5f94515668d7c23f5cb3f9233e3e9f3566793784095'
PROTECTED_EQUAL = ['assay_key', 'gene', 'assay_id', 'protein_reference_id', 'task', 'assay_type',
                   'score_kind', 'supervision_tier', 'is_direct_ndd', 'ndd_evidence_class',
                   'orientation', 'score_direction', 'source', 'loaded_rows',
                   'source_raw_phenotype', 'source_raw_directionality']
# Per-field status, the evidence ledger and the run stamp are written for every row; logging
# them as "changes" would bury the real content edits. They are in the researched table.
BOOKKEEPING = {'curation_evidence', 'curation_review_date', 'curation_schema_version'}

SEARCH_NOTES = {
    'original_v3': ('Checked the supplied corpus, README_ndd_training_corpus_v3.txt, '
                    'prepared/audit/assay_metadata.csv and the V4 build script for a '
                    'constituent-assay manifest or aggregation rule.'),
    'ProteinGym_v1.3': ('Checked the ProteinGym v1.3 reference table (selection_assay, '
                        'selection_type, raw_DMS_filename, raw phenotype column, region_mutated), '
                        'cached Europe PMC full text, MaveDB score-set and experiment records '
                        'found either by the URN ProteinGym names or by gene search matched on '
                        'publication DOI, and the publisher or preprint full text.'),
    'MaveDB_Domainome1': 'Checked the MaveDB score-set record (methodText, abstractText) and the study DOI.',
}

# Direction, target-definition and condition-mapping concerns found during curation. Reported,
# never silently applied: changing a sign or re-labelling a task is a measurement decision.
ISSUES = [
    ('CBS::CBS_HUMAN_Sun_2020', 'condition_mapping',
     'The study screened vitamin B6 at 0, 1 and 400 ng/mL. ProteinGym imports a column named only '
     '"score" and its raw_DMS_filename is blank, so the B6 arm cannot be traced.',
     'Do not attach a B6 concentration to this assay until the import is traced.'),
    ('PPM1D::PPM1D_HUMAN_Miller_2022', 'score_direction',
     'The reporter score is log2(GFP-high/GFP-low), where higher means *impaired* PPM1D '
     'phosphatase activity. ProteinGym labels the column "fitness" with directionality +1.',
     'Inspect the ProteinGym transformation before any biological reading of the sign. '
     'orientation and score_direction were left exactly as supplied.'),
    ('PPM1D::PPM1D_HUMAN_Miller_2022', 'condition_mapping',
     'Methods refer to sorting "from either treatment arm", implying a baseline arm alongside the '
     'daunorubicin arm, while a single fitness column was imported.',
     'The 200 nM / 24 h daunorubicin annotation applies only if the imported column is the treated '
     'arm; confirm against the raw supplementary workbook.'),
    ('PPARG::PPARG_HUMAN_Majithia_2016', 'target_definition',
     'raw_DMS_filename points at a MITER web query rather than a score table, and the study '
     'reports an integrated score built with variant classifications.',
     'Treat these scores as a model-integrated quantity, not an interchangeable raw measurement; '
     'using them as benchmark targets risks circularity with classification labels.'),
    ('SHOC2::SHOC2_HUMAN_Kwon_2022', 'condition_mapping',
     'The screen ran trametinib and DMSO arms; the imported column is LFC_scaled and the raw file '
     'is named "Extended Data Table 4", which was not locatable in the accessible text.',
     'Cell line (PA-TU-8902), dose (10 nM) and duration (16 days) are verified for the screen; '
     'whether LFC_scaled is the trametinib arm or a trametinib-vs-DMSO contrast is unconfirmed.'),
    ('DLG4::DLG4_HUMAN_Faure_2021', 'task_label',
     'ProteinGym imports a generic "fitness" column as OrganismalFitness, but the study has '
     'distinct abundancePCA and bindingPCA arms that both read out growth.',
     'Task, assay_type and measurement_type were left untouched. The five context fields are '
     'common to both arms and so were curated; the arm identity stays unresolved. Applies equally '
     'to GRB2_HUMAN_Faure_2021.'),
    ('DLG4::DLG4_HUMAN_Faure_2021', 'coordinate_mapping',
     'The same PSD95 PDZ3 domain is numbered 311-394 in the ProteinGym reference and 354-437 in '
     'the publication.',
     'region_coordinates states which numbering it uses; do not mix the two.'),
    ('GCK::HXK4_HUMAN_Gersing_2023_abundance', 'source_table_consistency',
     'ProteinGym lists the same raw_DMS_filename (HXK4_HUMAN_Gersing_2022.csv) and the same raw '
     'phenotype column ("score") for both GCK assays. The deposited MaveDB score sets show they '
     'are genuinely different experiments from different publications: activity by complementation '
     'of an hxk1/hxk2/glk1 triple deletion (DOI 10.1186/s13059-023-02935-8) and abundance by '
     'DHFR-PCA (DOI 10.1186/s13059-024-03238-2).',
     'The curated context follows the MaveDB score sets. Verify which score column ProteinGym '
     'actually imported for each before treating them as two independent properties.'),
    ('AICDA::AICDA_HUMAN_Gajula_2014_3cycles', 'source_table_wording',
     'The automatic enrichment matched the phrase "bulk RNA-sequencing" in the ProteinGym '
     'selection description and set system=bulk_RNA_sequencing. The assay is an E. coli rifampin '
     'selection read by 454 pyrosequencing of DNA.',
     'Corrected with MaveDB and paper evidence. Other assays whose system came from the same '
     'phrase matcher deserve the same check.'),
    ('KCNH2::KCNH2_HUMAN_Kozek_2020', 'source_table_wording',
     'ProteinGym describes the selection only as "Voltage". The assay measures cell-surface '
     'trafficking of HA-tagged KV11.1 by anti-HA staining and flow cytometry; no voltage is '
     'recorded.',
     'selection was set to surface_expression on the paper Methods. The task label was left '
     'untouched.'),
    ('KCNE1::KCNE1_HUMAN_Muhammad_2023_function', 'source_table_wording',
     'ProteinGym describes this as "potassium channel function". The deposited functional score '
     'set measures cell fitness in an LP-KCNQ1-S140G background over days, not conductance.',
     'selection was set to growth. Treat the score as a fitness proxy for channel function.'),
    ('KCNE1::KCNE1_HUMAN_Muhammad_2023_expression', 'condition_mapping',
     'The study deposited two trafficking score sets, in the presence (urn:mavedb:00000674-a-2) '
     'and absence (urn:mavedb:00000674-c-1) of KCNQ1. ProteinGym imports a single "TrafScore".',
     'Host, selection and system are common to both and were recorded; the KCNQ1 co-expression arm '
     'is unresolved.'),
    ('HMBS::HEM3_HUMAN_Loggerenberg_2023', 'condition_mapping',
     'The study deposited separate variant-effect maps for the ubiquitous (urn:mavedb:00000108-a-1) '
     'and erythroid-specific (-b-1) HMBS isoforms. ProteinGym imports one "score" column.',
     'Host, selection and system are common to both isoform maps and were recorded; which isoform '
     'the imported column holds is unresolved.'),
    ('CCR5::CCR5_HUMAN_Gill_2023', 'condition_mapping',
     'The study sorted on two gates, myc surface expression and BiFC signal, and ProteinGym '
     'describes the assay as "binding affinity, surface expression" with an "avg_score" column.',
     'Host, cell line and system were recorded; selection was deliberately left unstandardised.'),
    ('SRC::SRC_HUMAN_Nguyen_2022', 'score_semantics',
     'The imported "diffsel" column is a differential score: the calibrated radicicol activity '
     'score minus the DMSO activity score, not a single-arm measurement.',
     'radicicol is recorded with treatment_role=competition and the differential nature is noted '
     'in construct_notes. Do not read the value as an absolute activity.'),
    ('SRC::SRC_HUMAN_Ahler_2019', 'score_semantics',
     'The MaveDB record states that activity_score is -1 * score, because depletion in the '
     'population corresponds to higher kinase activity.',
     'Orientation left as supplied; check which convention ProteinGym imported before '
     'interpreting the sign.'),
    ('HMGCR::HMDH_HUMAN_Jiang_2019', 'target_definition',
     'The imported MaveDB set is titled "HMGCR rosuvastatin imputed and refined".',
     'Some values are imputed rather than directly measured; do not treat every row as an '
     'independent observation.'),
    ('BRCA1::BRCA1_HUMAN_Findlay_2018', 'region_definition',
     'ProteinGym records region_mutated as 1-1855 of 1863, which a coverage rule would read as '
     'whole-protein. Saturation genome editing actually targeted thirteen exons encoding the RING '
     'and BRCT domains (exons 2-5 and 15-23).',
     'region was set from the paper, not the coverage rule. Other assays whose region was derived '
     'only from region_mutated may carry the same over-statement.'),
    ('UBR5::UBR5_HUMAN_Tsuboyama_2023_1I2T', 'score_semantics',
     'The Tsuboyama raw phenotype column is named ddG_ML_float, which reads like a ddG, while the '
     'imported quantity is the folding stability the study fits.',
     'measurement_type is already proteolysis_stability; check the ProteinGym transformation '
     'before interpreting magnitudes or signs as ddG. Applies to all 22 Tsuboyama assays.'),
    ('ERBB2::ERBB2_HUMAN_Elazar_2016', 'score_semantics',
     'Elazar scores are apparent insertion free energies derived from selection coefficients at '
     '310 K, and ProteinGym records raw directionality -1.',
     'Applies equally to GLPA and LYAM1. Orientation left as supplied.'),
    ('CD19::CD19_HUMAN_Klesmith_2019_FMC_singles', 'citation_mismatch',
     'The metadata DOI (10.1021/acs.molpharmaceut.9b00418) is the CD19-fusion retargeting paper, '
     'while the single-site FMC63 epitope library the imported file names is reported in the '
     'companion Biochemistry paper (10.1021/acs.biochem.9b00808).',
     'Context was recorded at source_table_derived status only. Confirm which publication the '
     'score table belongs to before raising its status.'),
]

CONSENSUS_REQUIREMENTS = [
    'constituent_assay_identifiers',
    'raw_phenotype_column_per_constituent_assay',
    'score_transform_and_sign_per_constituent_assay',
    'aggregation_weights_or_rules',
    'construct_and_coordinate_mapping_per_constituent_assay',
]


def changes_table(before, after, document):
    """One row per changed content field, each carrying its own evidence."""
    overrides = document['overrides']
    rows = []
    for column in after.columns:
        if column in BOOKKEEPING or column.endswith('_status'):
            continue
        old_series = (before[column].astype(str) if column in before.columns
                      else pd.Series(['']*len(before), index=before.index))
        new_series = after[column].astype(str)
        for index in after.index:
            old, new = old_series.loc[index], new_series.loc[index]
            if old == new:
                continue
            key = after.at[index, 'assay_key']
            record = overrides.get(key, {}).get('fields', {}).get(column)
            rows.append({'assay_key': key, 'field': column, 'old_value': old, 'new_value': new,
                         'source_url': record['source_url'] if record else '',
                         'locator': record['locator'] if record else '',
                         'evidence_summary': record['evidence_summary'] if record else
                                             'Derived column added by the curation pass.',
                         'evidence_scope': record['evidence_scope'] if record else 'derived',
                         'review_date': document['review_date']})
    frame = pd.DataFrame(rows)
    return frame.sort_values(['assay_key', 'field']).reset_index(drop=True) if len(frame) else frame


def unresolved_table(after):
    """One row per model field still unresolved, with why and what would settle it."""
    rows = []
    for row in after.itertuples():
        for field in MODEL_FIELDS:
            value = getattr(row, field)
            if value not in ('', 'unknown'):
                continue
            if row.source == 'original_v3':
                why = ('Collapsed consensus target: the supplied files contain no constituent assay '
                       'identifiers, raw phenotype columns, score transforms or aggregation rules, '
                       'so no experimental context can be attributed to this row.')
                need = ('Original consensus build script or source manifest listing constituent '
                        'assay IDs, their raw phenotype columns, score transforms/signs, '
                        'aggregation weights and construct mapping.')
            elif field == 'treatment':
                why = ('No treatment arm for this assay was established in the reviewed sources. '
                       'Left unknown rather than labelled untreated, because no explicitly '
                       'untreated arm was documented either.')
                need = ('Primary Methods section or deposited protocol naming the scored arm and '
                        'its perturbation.')
            else:
                why = 'Not established for this exact assay in the sources reviewed.'
                need = ('Primary full text (Methods, figure legends, supplementary score-table '
                        'header) or a deposited MaveDB score set for this assay.')
            rows.append({'assay_key': row.assay_key, 'gene': row.gene, 'source': row.source,
                         'is_direct_ndd': row.is_direct_ndd, 'field': field,
                         'current_value': value or 'unknown',
                         'status': getattr(row, field+'_status'),
                         'what_was_searched': SEARCH_NOTES.get(row.source, ''),
                         'why_unresolved': why, 'next_source_needed': need})
    return pd.DataFrame(rows)


def consensus_table(after):
    rows = []
    for row in after[after.source == 'original_v3'].itertuples():
        for item in CONSENSUS_REQUIREMENTS:
            rows.append({'assay_key': row.assay_key, 'gene': row.gene,
                         'loaded_rows': row.loaded_rows, 'missing_provenance_item': item,
                         'searched': SEARCH_NOTES['original_v3'],
                         'status': 'original_provenance_missing',
                         'note': 'Measurement values, identifiers and orientation=as_supplied are '
                                 'retained unchanged; no experimental context was attributed.'})
    return pd.DataFrame(rows)


def baseline_frame():
    """Return the pre-curation table, snapshotting it once so the change log stays stable."""
    CURATION.mkdir(parents=True, exist_ok=True)
    snapshot = CURATION/'assay_metadata_baseline.csv'
    if not snapshot.exists():
        if digest(PATH) != BASELINE_SHA:
            print('NOTE: no baseline snapshot exists and the current metadata does not match the '
                  'recorded baseline SHA-256; the change log will be relative to this file.')
        snapshot.write_bytes(PATH.read_bytes())
    return pd.read_csv(snapshot, keep_default_na=False)


def main():
    before = baseline_frame()
    document = load_overrides()

    frame = enrich_metadata(before)                     # automatic, conservative
    frame = apply_overrides(frame, document=document)   # curated evidence wins, and is replayable
    frame = default_status(frame)

    for column in PROTECTED_EQUAL:
        if not frame[column].astype(str).equals(before[column].astype(str)):
            raise AssertionError(f'{column} changed during a metadata-only curation pass')
    if len(frame) != len(before) or frame.assay_key.duplicated().any():
        raise AssertionError('assay row set changed')

    frame.to_csv(PATH, index=False)
    frame.to_csv(CURATION/'assay_metadata_researched.csv', index=False)
    changes = changes_table(before, frame, document)
    changes.to_csv(CURATION/'metadata_changes.csv', index=False)
    unresolved = unresolved_table(frame)
    unresolved.to_csv(CURATION/'metadata_unresolved.csv', index=False)
    issues = pd.DataFrame([{'assay_key': k, 'issue_type': t, 'observation': o,
                            'recommended_handling': h, 'review_date': document['review_date']}
                           for k, t, o, h in ISSUES])
    missing = sorted(set(issues.assay_key)-set(frame.assay_key))
    if missing:
        raise AssertionError(f'issue rows reference unknown assays: {missing}')
    issues.to_csv(CURATION/'metadata_issues.csv', index=False)
    consensus_table(frame).to_csv(CURATION/'consensus_provenance_required.csv', index=False)

    def counts(table):
        return {f: int(table[f].isin(['', 'unknown']).sum()) for f in MODEL_FIELDS}

    report = {'assays': len(frame), 'metadata_sha256': digest(PATH), 'baseline_sha256': BASELINE_SHA,
              'unknown_by_field_before': counts(before), 'unknown_by_field_after': counts(frame),
              'changed_cells': {f: int((frame[f].astype(str) != before[f].astype(str)).sum())
                                for f in MODEL_FIELDS},
              'field_status_counts': {f: frame[f+'_status'].value_counts().to_dict()
                                      for f in MODEL_FIELDS},
              'changed_field_rows': len(changes), 'unresolved_field_rows': len(unresolved),
              'assays_with_overrides': len(document['overrides']),
              'issues_reported': len(issues),
              'source_groups': frame.groupby('source').size().to_dict(),
              'review_date': document['review_date']}
    save_json('v4/audit/metadata_coverage.json', report)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
