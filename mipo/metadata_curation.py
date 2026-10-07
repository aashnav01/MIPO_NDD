"""Conservative source-backed enrichment; never infer host from target species."""
import json
from pathlib import Path

OVERRIDES = Path('v4/data/assay_context_overrides.json')
# Columns a curated override may never touch: identity, measurement semantics and provenance.
PROTECTED = {'gene', 'assay_id', 'assay_key', 'protein_reference_id', 'task', 'assay_type',
             'score_kind', 'supervision_tier', 'taxon_id', 'orientation', 'score_direction',
             'is_direct_ndd', 'ndd_evidence_class', 'source', 'source_url', 'source_sha256',
             'measurement_type', 'loaded_rows', 'source_raw_phenotype', 'source_raw_directionality',
             'selection_assay', 'selection_type', 'publication_doi', 'license'}
MODEL_FIELDS = ['host', 'selection', 'system', 'treatment', 'region']


def load_overrides(path=OVERRIDES):
    """Read the curated override document, or return an empty one when it is absent."""
    path = Path(path)
    if not path.exists():
        return {'allowed_fields': [], 'overrides': {}, 'review_date': '', 'schema_version': ''}
    document = json.loads(path.read_text(encoding='utf-8'))
    forbidden = set(document.get('allowed_fields', [])) & PROTECTED
    if forbidden:
        raise ValueError(f'override file allows protected columns: {sorted(forbidden)}')
    return document


def apply_overrides(frame, path=OVERRIDES, document=None):
    """Replay curated, source-backed context onto exact assay_key matches.

    Runs *after* ``enrich_metadata`` so that automatic re-extraction cannot overwrite curated
    values on a rebuild. Matching is on the full ``assay_key`` only -- never on gene, DOI or
    any other group -- so a study-level finding cannot leak into an unrelated assay. Applying
    the same document twice produces the same table.
    """
    document = document if document is not None else load_overrides(path)
    overrides = document.get('overrides', {})
    allowed = set(document.get('allowed_fields', []))
    frame = frame.copy().fillna('')
    known = set(frame.get('assay_key', []))
    for field in MODEL_FIELDS:
        if field+'_status' not in frame:
            # Everything starts un-reviewed; the loop below upgrades what evidence covers.
            frame[field+'_status'] = ''
    ledger = {}
    for assay_key, entry in overrides.items():
        if assay_key not in known:
            continue
        mask = frame.assay_key == assay_key
        for field, record in entry.get('fields', {}).items():
            if field not in allowed:
                raise ValueError(f'{assay_key}: field {field!r} is outside allowed_fields')
            if field in PROTECTED:
                raise ValueError(f'{assay_key}: field {field!r} is protected')
            if field not in frame:
                frame[field] = ''
            frame.loc[mask, field] = record['value']
            if field in MODEL_FIELDS:
                frame.loc[mask, field+'_status'] = record['status']
            ledger.setdefault(assay_key, {})[field] = {
                'status': record['status'], 'source_url': record['source_url'],
                'locator': record['locator'], 'evidence_scope': record['evidence_scope']}
    if 'curation_evidence' not in frame:
        frame['curation_evidence'] = ''
    frame['curation_evidence'] = [json.dumps(ledger[key], sort_keys=True) if key in ledger else old
                                  for key, old in zip(frame.assay_key, frame['curation_evidence'])]
    frame['curation_review_date'] = document.get('review_date', '')
    frame['curation_schema_version'] = document.get('schema_version', '')
    return frame


def default_status(frame):
    """Label the model fields that no curated evidence reached, so gaps stay visible."""
    frame = frame.copy().fillna('')
    for field in MODEL_FIELDS:
        column = field+'_status'
        if column not in frame:
            frame[column] = ''
        blank = frame[column] == ''
        unknown = frame[field].isin(['', 'unknown'])
        consensus = frame['source'] == 'original_v3'
        frame.loc[blank & consensus & unknown, column] = 'original_provenance_missing'
        frame.loc[blank & ~consensus & unknown, column] = 'not_reported_in_reviewed_sources'
        frame.loc[blank & ~unknown, column] = 'source_table_derived'
    return frame


def enrich_metadata(frame):
    frame = frame.copy().fillna('')
    for index, row in frame.iterrows():
        evidence = {}
        if row['source'] == 'ProteinGym_v1.3':
            assay = str(row.get('selection_assay', '')).strip()
            kind = str(row.get('selection_type', '')).strip()
            text = (assay+' '+kind).lower()
            if assay:
                frame.at[index, 'selection'] = assay
                evidence['selection'] = 'ProteinGym reference: selection_assay (verbatim)'
            if 'yeast' in text:
                frame.at[index, 'host'] = 'yeast'
                evidence['host'] = 'Explicit yeast mention in ProteinGym selection description'
            methods = [('cdna display proteolysis', 'cDNA_display_proteolysis'),
                       ('phage display', 'phage_display'), ('yeast-displayed', 'yeast_display'),
                       ('flow cytometry', 'flow_cytometry'), ('facs', 'FACS'),
                       ('bulk rna-sequencing', 'bulk_RNA_sequencing')]
            for phrase, value in methods:
                if phrase in text:
                    frame.at[index, 'system'] = value
                    evidence['system'] = 'Explicit method in ProteinGym selection_assay/selection_type'
                    break
            # Only copy drugs named explicitly; absence does not mean untreated.
            drugs = [drug for drug in ['trametinib', 'ibrutinib', 'isoproterenol',
                     'veratridine', 'brevetoxin', 'ouabain'] if drug in text]
            if drugs:
                frame.at[index, 'treatment'] = '|'.join(drugs)
                evidence['treatment'] = 'Explicit named compound(s) in ProteinGym selection description; dose unspecified'
            status = 'partial_source_backed_context'
            reason = 'Remaining unknown fields are not explicitly recoverable by this reference-table mapping; primary-study curation remains necessary'
        elif row['source'] == 'original_v3':
            status = 'unresolved_collapsed_consensus'
            reason = 'Original constituent assays, protocols and biological score direction were not supplied; do not borrow context from another assay of the same gene'
        else:
            status = 'study_level_context'
            reason = 'Shared Domainome protocol; see publication and score-set source metadata'
            evidence = {'context': 'Domainome study protocol; source DOI and score-set URL recorded in this row'}
        frame.at[index, 'context_status'] = status
        frame.at[index, 'context_limitations'] = reason
        frame.at[index, 'context_evidence'] = json.dumps(evidence, sort_keys=True)
    return frame
