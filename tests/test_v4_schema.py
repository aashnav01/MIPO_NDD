import json

import pandas as pd
import pytest

from mipo.corpus import read_corpus
from mipo.smoke import fixture
from mipo.splits import make_splits, verify_fold


def test_separate_construct_numbering_same_gene(tmp_path):
    paths = fixture(tmp_path)
    d = pd.read_csv(paths[0])
    d['protein_reference_id'] = d.gene+'__full'
    domain = d[d.gene == 'SYNTHETIC_0'].iloc[:1].copy()
    domain['protein_reference_id'] = 'SYNTHETIC_0__domain'
    domain['assay_id'] = 'domain_stability'
    domain['assay_type'] = 'stability'
    domain['score_kind'] = 'MaveDB_domainome_normalized_score'
    domain['wt'] = 'A' if domain.wt.iloc[0] != 'A' else 'C'
    if domain.mut.iloc[0] == domain.wt.iloc[0]:
        domain['mut'] = 'W'
    pd.concat([d, domain], ignore_index=True).to_csv(tmp_path/'multi.csv', index=False)
    result = read_corpus(tmp_path/'multi.csv')
    assert result.protein_reference_id.nunique() == 6
    assert result.gene.nunique() == 5
    splits = make_splits(tmp_path/'multi.csv', tmp_path/'multi_splits.json')
    for fold in splits['folds']:
        verify_fold(result, fold)
        if 'SYNTHETIC_0' in fold['test']:
            assert set(result[result.gene.isin(fold['test'])].protein_reference_id) == {'SYNTHETIC_0__full', 'SYNTHETIC_0__domain'}


def test_ndd_outer_targets_keep_generic_teachers_in_training(tmp_path):
    paths = fixture(tmp_path)
    target_file = tmp_path/'ndd.csv'
    pd.DataFrame({'gene': ['SYNTHETIC_0', 'SYNTHETIC_1', 'SYNTHETIC_2']}).to_csv(target_file, index=False)
    splits = make_splits(paths[0], tmp_path/'target_splits.json', test_genes=target_file)
    assert len(splits['folds']) == 3
    for fold in splits['folds']:
        assert {'SYNTHETIC_3', 'SYNTHETIC_4'} <= set(fold['train'])
        assert not {'SYNTHETIC_3', 'SYNTHETIC_4'} & set(fold['test']+fold['validation']+fold['calibration'])


def test_same_reference_cannot_belong_to_different_genes(tmp_path):
    paths = fixture(tmp_path)
    d = pd.read_csv(paths[0])
    d['protein_reference_id'] = 'bad_shared_id'
    d.to_csv(tmp_path/'bad.csv', index=False)
    with pytest.raises(ValueError):
        read_corpus(tmp_path/'bad.csv')
