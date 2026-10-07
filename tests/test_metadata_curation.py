from pathlib import Path

import pandas as pd
import pytest

from mipo.metadata_curation import (MODEL_FIELDS, PROTECTED, apply_overrides, default_status,
                                    enrich_metadata, load_overrides)

BASE = dict(host='unknown', selection='unknown', system='unknown', treatment='unknown',
            region='unknown', score_direction='as_supplied_unknown_biological_direction',
            orientation='as_supplied', loaded_rows=10)


def field(value, status='verified_assay_specific'):
    return {'value': value, 'status': status, 'source_url': 'u', 'locator': 'l',
            'evidence_summary': 's', 'evidence_scope': 'assay_specific'}


def document(overrides, allowed=('host', 'treatment', 'region')):
    return {'schema_version': 't', 'review_date': '2026-09-20',
            'allowed_fields': list(allowed), 'overrides': overrides}


def test_context_requires_explicit_source_evidence():
    rows = pd.DataFrame([
        dict(BASE, source='ProteinGym_v1.3', selection_assay='Yeast growth', selection_type='Growth'),
        dict(BASE, source='ProteinGym_v1.3', selection_assay='Drug resistance', selection_type='Survival (dosed with trametinib)'),
        dict(BASE, source='original_v3', selection_assay='Yeast growth', selection_type='Growth'),
    ])
    result = enrich_metadata(rows)
    assert result.iloc[0].host == 'yeast'
    assert result.iloc[0].treatment == 'unknown'
    assert result.iloc[1].host == 'unknown'
    assert result.iloc[1].treatment == 'trametinib'
    assert result.iloc[2].host == 'unknown'
    assert result.iloc[2].context_status == 'unresolved_collapsed_consensus'


def test_overrides_match_the_exact_assay_only():
    """A finding for one assay must not reach another assay of the same gene."""
    rows = pd.DataFrame([
        dict(BASE, assay_key='PTEN::PTEN_HUMAN_Matreyek_2021', gene='PTEN', source='ProteinGym_v1.3'),
        dict(BASE, assay_key='PTEN::PTEN_HUMAN_Mighell_2018', gene='PTEN', source='ProteinGym_v1.3'),
        dict(BASE, assay_key='PTEN::gate_consensus', gene='PTEN', source='original_v3'),
    ])
    result = apply_overrides(rows, document=document(
        {'PTEN::PTEN_HUMAN_Matreyek_2021': {'fields': {'host': field('human_cell_line')}}}))
    assert result.iloc[0].host == 'human_cell_line'
    assert result.iloc[1].host == 'unknown'     # same gene, different assay: untouched
    assert result.iloc[2].host == 'unknown'     # legacy consensus never inherits context
    assert result.iloc[0].host_status == 'verified_assay_specific'
    assert result.iloc[1].host_status == ''


def test_unknown_assay_keys_are_ignored_not_applied_by_position():
    rows = pd.DataFrame([dict(BASE, assay_key='A::a', gene='A', source='ProteinGym_v1.3')])
    result = apply_overrides(rows, document=document({'B::b': {'fields': {'host': field('yeast')}}}))
    assert result.iloc[0].host == 'unknown'


def test_protected_columns_cannot_be_overridden():
    rows = pd.DataFrame([dict(BASE, assay_key='A::a', gene='A', source='ProteinGym_v1.3')])
    bad = document({'A::a': {'fields': {'score_direction': field('higher_is_healthier')}}},
                   allowed=('score_direction',))
    with pytest.raises(ValueError):
        apply_overrides(rows, document=bad)


def test_field_outside_allowed_list_is_rejected():
    rows = pd.DataFrame([dict(BASE, assay_key='A::a', gene='A', source='ProteinGym_v1.3')])
    bad = document({'A::a': {'fields': {'region': field('full_length')}}}, allowed=('host',))
    with pytest.raises(ValueError):
        apply_overrides(rows, document=bad)


def test_replay_is_idempotent_and_survives_re_enrichment():
    """Rebuilds re-run enrich_metadata; curated values must come back unchanged."""
    rows = pd.DataFrame([dict(BASE, assay_key='AICDA::x', gene='AICDA', source='ProteinGym_v1.3',
                              selection_assay='Enzymatic activity',
                              selection_type='bulk RNA-sequencing')])
    doc = document({'AICDA::x': {'fields': {'host': field('e_coli')}}})
    once = apply_overrides(enrich_metadata(rows), document=doc)
    twice = apply_overrides(enrich_metadata(once), document=doc)
    assert once.iloc[0].host == 'e_coli'
    assert twice.iloc[0].host == 'e_coli'
    assert once[['host', 'host_status']].equals(twice[['host', 'host_status']])


def test_default_status_labels_gaps_without_filling_them():
    rows = pd.DataFrame([
        dict(BASE, assay_key='A::a', gene='A', source='ProteinGym_v1.3'),
        dict(BASE, assay_key='B::gate_consensus', gene='B', source='original_v3'),
    ])
    result = default_status(rows)
    assert result.iloc[0].host == 'unknown'          # value untouched
    assert result.iloc[0].host_status == 'not_reported_in_reviewed_sources'
    assert result.iloc[1].host_status == 'original_provenance_missing'


def test_curated_document_is_wellformed_and_matches_the_real_table():
    """The shipped override file must only name real assays and permitted fields."""
    path, metadata = Path('v4/data/assay_context_overrides.json'), Path('v4/data/assay_metadata.csv')
    if not (path.exists() and metadata.exists()):
        pytest.skip('curated override file or V4 metadata not present')
    doc = load_overrides(path)
    keys = set(pd.read_csv(metadata, keep_default_na=False).assay_key)
    allowed = set(doc['allowed_fields'])
    assert not allowed & PROTECTED
    for assay_key, entry in doc['overrides'].items():
        assert assay_key in keys, assay_key
        for name, record in entry['fields'].items():
            assert name in allowed, (assay_key, name)
            for required in ['value', 'status', 'source_url', 'locator', 'evidence_summary',
                             'evidence_scope']:
                assert str(record.get(required, '')).strip(), (assay_key, name, required)


def test_curated_metadata_preserves_measurement_semantics():
    """The curated table must not have moved a score, an identifier or an orientation."""
    current, baseline = Path('v4/data/assay_metadata.csv'), Path('v4/curation/assay_metadata_baseline.csv')
    if not (current.exists() and baseline.exists()):
        pytest.skip('curation has not been run in this checkout')
    after = pd.read_csv(current, keep_default_na=False)
    before = pd.read_csv(baseline, keep_default_na=False)
    assert len(after) == len(before) == 625
    assert not after.assay_key.duplicated().any()
    for column in ['assay_key', 'gene', 'assay_id', 'protein_reference_id', 'task', 'assay_type',
                   'score_kind', 'supervision_tier', 'is_direct_ndd', 'orientation',
                   'score_direction', 'source', 'loaded_rows', 'source_raw_directionality']:
        assert after[column].astype(str).equals(before[column].astype(str)), column
    consensus = after[after.source == 'original_v3']
    assert len(consensus) == 25
    # Consensus context is allowed only where a source assay was identified by rank matching.
    # Everything else must still be unknown with the provenance-missing status.
    if 'recovered_source_urn' not in after.columns:
        pytest.skip('consensus recovery has not been applied to this table')
    urn = consensus.recovered_source_urn.astype(str)
    recovered, unrecovered = consensus[urn != ''], consensus[urn == '']
    for name in MODEL_FIELDS:
        assert set(unrecovered[name]) <= {'unknown', ''}, name
        assert set(unrecovered[name+'_status']) == {'original_provenance_missing'}, name
    assert len(recovered) == 13, len(recovered)


def test_recovered_consensus_carries_its_rank_evidence():
    """A consensus row may only gain context if it names the source and the match strength."""
    current = Path('v4/data/assay_metadata.csv')
    if not current.exists():
        pytest.skip('V4 metadata not present')
    after = pd.read_csv(current, keep_default_na=False)
    if 'recovered_source_urn' not in after:
        pytest.skip('consensus recovery has not been run')
    recovered = after[(after.source == 'original_v3') & (after.recovered_source_urn.astype(str) != '')]
    for row in recovered.itertuples():
        assert row.recovered_source_urn.startswith('urn:mavedb:'), row.assay_key
        rho = float(row.recovered_rank_correlation)
        assert abs(rho) > 0.99, (row.assay_key, rho)
        assert row.recovered_orientation in ('preserved_relative_to_source',
                                             'inverted_relative_to_source'), row.assay_key
        # A negative rank correlation must be declared, never quietly absorbed.
        assert (rho < 0) == (row.recovered_orientation == 'inverted_relative_to_source'), row.assay_key


def test_no_assay_gained_an_undocumented_value():
    """Every non-unknown model field carries a status, and every verified status has a value."""
    current = Path('v4/data/assay_metadata.csv')
    if not current.exists():
        pytest.skip('V4 metadata not present')
    after = pd.read_csv(current, keep_default_na=False)
    for name in MODEL_FIELDS:
        assert (after[name+'_status'] != '').all(), name
        verified = after[name+'_status'].isin(
            ['verified_assay_specific', 'verified_shared_protocol', 'not_applicable'])
        assert not after.loc[verified, name].isin(['', 'unknown']).any(), name


def test_not_applicable_is_a_value_not_a_missingness_label():
    """not_applicable must be backed by evidence, never used to hide an unknown."""
    current = Path('v4/data/assay_metadata.csv')
    if not current.exists():
        pytest.skip('V4 metadata not present')
    after = pd.read_csv(current, keep_default_na=False)
    for name in MODEL_FIELDS:
        flagged = after[after[name+'_status'] == 'not_applicable']
        assert (flagged[name] == 'not_applicable').all(), name
        # and the converse: the literal value is only ever used with that status
        holds = after[after[name] == 'not_applicable']
        assert set(holds[name+'_status']) <= {'not_applicable', 'verified_shared_protocol',
                                              'verified_assay_specific'}, name
