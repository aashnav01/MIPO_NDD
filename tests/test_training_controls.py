import importlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from mipo.common import save_json
from mipo.comparison import matched_llr_comparison
from mipo.corpus import read_corpus
from mipo.data import FeatureStore
from mipo.metrics import add_intervals, metrics
from mipo.smoke import fixture
from mipo.splits import expand_holdouts, make_splits, verify_fold
from mipo.train import selection_score
from scripts.run_experiments import completed_run_matches


def test_expansion_preserves_outer_roles_and_is_seed_independent_of_training(tmp_path):
    paths = fixture(tmp_path)
    original = json.loads(paths[3].read_text())
    first = expand_holdouts(paths[0], paths[3], tmp_path/'expanded.json', 2, 1, 17)
    again = expand_holdouts(paths[0], paths[3], tmp_path/'again.json', 2, 1, 17)
    assert first == again
    assert json.loads(paths[3].read_text()) == original
    data = read_corpus(paths[0])
    for before, after in zip(original['folds'], first['folds']):
        assert after['test'] == before['test']
        assert set(before['validation']) <= set(after['validation'])
        assert after['calibration'] == before['calibration']
        assert len(after['validation']) == 2
        assert len(after['train']) == 1
        verify_fold(data, after, first['gene_to_group'])


def test_expansion_keeps_clusters_and_rejects_impossible_panel(tmp_path):
    paths = fixture(tmp_path)
    genes = sorted(read_corpus(paths[0]).gene.unique())
    clusters = tmp_path/'clusters.csv'
    pd.DataFrame({'gene': genes, 'cluster': ['a', 'a', 'b', 'c', 'd']}).to_csv(clusters, index=False)
    make_splits(paths[0], tmp_path/'clustered.json', clusters)
    result = expand_holdouts(paths[0], tmp_path/'clustered.json', tmp_path/'expanded.json', 1, 1)
    for fold in result['folds']:
        for role in ['train', 'validation', 'calibration', 'test']:
            assert (genes[0] in fold[role]) == (genes[1] in fold[role])
    with pytest.raises(ValueError, match='insufficient eligible'):
        expand_holdouts(paths[0], tmp_path/'clustered.json', tmp_path/'impossible.json', 2, 1)
    fold = result['folds'][0]
    other = next(r for r in ['validation', 'calibration', 'train'] if genes[0] not in fold[r])
    original_role = next(r for r in ['test', 'validation', 'calibration', 'train'] if genes[0] in fold[r])
    fold[original_role].remove(genes[0])
    fold[other].append(genes[0])
    with pytest.raises(ValueError, match='Cluster leakage'):
        verify_fold(read_corpus(paths[0]), fold, result['gene_to_group'])


def predictions():
    return pd.DataFrame({'gene': ['large']*30+['small']*3,
                         'assay_key': ['large::a']*30+['small::a']*3,
                         'task': ['consensus']*30+['activity']*3,
                         'y': list(range(30))+[0, 1, 2],
                         'mu': list(range(30))+[2, 1, 0],
                         'sigma': [1.]*33, 'task_seen_in_training': [True]*33})


def test_selection_is_equal_gene_weighted_and_requires_all_genes():
    _, summary = metrics(predictions())
    assert selection_score(summary, 2) == pytest.approx(0.)
    data = predictions()
    data.loc[data.gene == 'small', 'mu'] = 0.
    _, summary = metrics(data)
    with pytest.raises(ValueError, match='one or more genes'):
        selection_score(summary, 2)


def test_gaussian_and_calibrated_intervals_are_distinct_and_missing_stays_missing():
    data = predictions()
    data['mu'] = data.y + 2
    calibrated = add_intervals(data, {'tasks': {'consensus': {'q': 3.}}})
    assays, summary = metrics(calibrated)
    assert assays.coverage_90_gaussian.eq(0).all()
    assert assays.loc[assays.task == 'consensus', 'coverage_calibrated'].iloc[0] == 1.
    assert pd.isna(assays.loc[assays.task == 'activity', 'coverage_calibrated'].iloc[0])
    assert summary['rows_without_calibrated_intervals'] == 3
    assert summary['per_task']['activity']['macro_gene_coverage_calibrated'] is None
    assert assays.bias.eq(2).all()
    assert assays.mean_sigma.eq(1).all()


def test_exact_variant_llr_comparison_and_missing_cache_rejected(tmp_path):
    paths = fixture(tmp_path)
    corpus = read_corpus(paths[0])
    store = FeatureStore(paths[1], paths[2], allow_synthetic=True)
    rows = corpus[corpus.gene == 'SYNTHETIC_0'].copy()
    rows['mu'] = rows.score_value
    assays, summary = matched_llr_comparison(rows[['row_id', 'gene', 'assay_key', 'variant_key', 'task', 'score_value', 'mu']], corpus, store)
    assert summary['matched_rows'] == len(rows)
    assert summary['model_macro_gene_spearman'] == pytest.approx(1.)
    bad = rows.copy()
    bad.loc[bad.index[0], 'score_value'] = -999
    with pytest.raises(ValueError, match='exactly match'):
        matched_llr_comparison(bad[['row_id', 'gene', 'assay_key', 'variant_key', 'task', 'score_value', 'mu']], corpus, store)
    store.protein('SYNTHETIC_0')['llr'] = None
    with pytest.raises(ValueError, match='every test row'):
        matched_llr_comparison(rows[['row_id', 'gene', 'assay_key', 'variant_key', 'task', 'score_value', 'mu']], corpus, store)


def test_completed_run_does_not_silently_skip_changed_inputs(tmp_path):
    save_json(tmp_path/'test_summary.json', {})
    save_json(tmp_path/'provenance.json', {'config': {'seed': 42}, 'split_sha256': 'old'})
    with pytest.raises(ValueError, match='Completed run inputs'):
        completed_run_matches(tmp_path, {'seed': 42}, {'split_sha256': 'new'})
    with pytest.raises(ValueError, match='Completed run inputs'):
        completed_run_matches(tmp_path, {'seed': 123}, {'split_sha256': 'old'})
    assert not completed_run_matches(tmp_path, {'seed': 42}, {'split_sha256': 'old'})


def test_multigene_training_diagnostics_and_resume_keeps_early_stop(tmp_path, monkeypatch):
    trainer = importlib.import_module('mipo.train')
    paths = fixture(tmp_path)
    expand_holdouts(paths[0], paths[3], tmp_path/'multi.json', 2, 1)
    config = json.loads(paths[-1].read_text())
    config.update(epochs=5, patience=1, min_validation_genes=2)
    save_json(paths[-1], config)
    original_metrics = trainer.metrics
    def constant_selection(data):
        assays, summary = original_metrics(data)
        summary['macro_gene_spearman'] = .25
        return assays, summary
    monkeypatch.setattr(trainer, 'metrics', constant_selection)
    run = tmp_path/'run'
    trainer.train(*paths[:3], tmp_path/'multi.json', paths[-1], out=run, allow_synthetic=True)
    before = torch.load(run/'last.pt', weights_only=True)
    assert before['epoch'] == 1
    assert json.loads((run/'training_status.json').read_text())['best_epoch'] == 0
    history = json.loads((run/'history.json').read_text())
    assert len(history[0]['validation_per_gene']) == 2
    assert set(history[0]['loss_components']) == {'nll', 'huber', 'ranking', 'field', 'auxiliary'}
    assert (run/'validation_epochs/epoch_001.csv').exists()
    assert {'train', 'validation', 'calibration', 'test'} == set(pd.read_csv(run/'split_composition.csv').role)
    test = pd.read_csv(run/'test_predictions.csv')
    assert np.allclose(test.prediction_score_units, test.mu)  # consensus uses identity scaling
    assert np.allclose(test.sigma_score_units, test.sigma)
    def unexpected_training(*args, **kwargs):
        raise AssertionError('Early-stopped run must not take another training step')
    monkeypatch.setattr(trainer, 'loss_function', unexpected_training)
    trainer.train(*paths[:3], tmp_path/'multi.json', paths[-1], out=run, resume=True, allow_synthetic=True)
    after = torch.load(run/'last.pt', weights_only=True)
    assert after['history'] == before['history']
    assert all(torch.equal(value, after['model'][key]) for key, value in before['model'].items())
