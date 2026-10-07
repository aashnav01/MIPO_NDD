import importlib
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mipo.common import save_json
from mipo.corpus import read_corpus
from mipo.data import FeatureStore, Metadata, TargetTransform
from mipo.features import cache_status
from mipo.smoke import fixture
from mipo.splits import expand_holdouts
from mipo.train import loader

ROOT = Path(__file__).resolve().parents[1]


def test_cache_status_rejects_manifest_only_and_partial_features(tmp_path):
    corpus, resources, cache = fixture(tmp_path)[:3]
    manifest = json.loads((cache/'manifest.json').read_text())
    save_json(cache/'manifest.json', {**manifest, 'llr': 'masked_marginal'})
    assert cache_status(corpus, resources, cache)['complete']
    interrupted = tmp_path/'interrupted'
    interrupted.mkdir()
    shutil.copy(cache/'manifest.json', interrupted/'manifest.json')
    status = cache_status(corpus, resources, interrupted)
    assert not status['complete'] and len(status['missing_embeddings']) == 5 and status['missing_llr_sites'] == 40
    records = json.loads((resources/'sequences.json').read_text())
    folder = cache/records['SYNTHETIC_0']['key']
    llr = np.load(folder/'llr.npy')
    llr[0, 3] = np.nan
    np.save(folder/'llr.npy', llr)
    (cache/records['SYNTHETIC_1']['key']/'sequence.json').unlink()
    status = cache_status(corpus, resources, cache)
    assert status['missing_llr_sites'] == 1 and status['missing_embeddings'] == ['SYNTHETIC_1']
    assert cache_status(corpus, resources, cache, llr=False)['missing_llr_sites'] == 0
    assert cache_status(corpus, resources, cache, model_name='facebook/esm2_t33_650M_UR50D')['incompatible']


def test_worker_processes_change_speed_not_batches(tmp_path):
    paths = fixture(tmp_path)
    d = read_corpus(paths[0])
    config = json.loads(paths[-1].read_text())
    store = FeatureStore(paths[1], paths[2], allow_synthetic=True)
    transform, metadata = TargetTransform().fit(d), Metadata().fit(d)
    for training in [True, False]:
        serial = list(loader(d, store, transform, metadata, config, training, 7, workers=0))
        parallel = list(loader(d, store, transform, metadata, config, training, 7, workers=2))
        assert len(serial) == len(parallel) > 0
        for a, b in zip(serial, parallel):
            assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_model_comparison_uses_only_seed_folds_every_model_finished():
    launcher = importlib.import_module('scripts.dgx_v4_validation')
    table = pd.DataFrame({'model': ['mipo', 'mipo', 'esm_mlp'], 'seed': [42, 42, 42], 'fold': [0, 1, 0],
                          'gene': ['A', 'B', 'A'], 'model_spearman': [.5, .9, .4], 'llr_spearman': [.3, .3, .3]})
    keep, common, dropped = launcher.common_evaluation(table, ['mipo', 'esm_mlp'])
    assert common == {(42, 0)} and set(keep.gene) == {'A'} and dropped == {'mipo': [(42, 1)]}
    assert launcher.common_evaluation(table, ['mipo', 'no_structure'])[0].empty


def test_local_cache_mirror_copies_new_and_changed_files_only(tmp_path):
    launcher = importlib.import_module('scripts.dgx_v4_validation')
    source, target = tmp_path/'drive', tmp_path/'local'
    (source/'a').mkdir(parents=True)
    (source/'a'/'esm.npy').write_bytes(b'one')
    (source/'a'/'llr.tmp.npy').write_bytes(b'partial')
    launcher.mirror(source, target)
    assert (target/'a'/'esm.npy').read_bytes() == b'one' and not (target/'a'/'llr.tmp.npy').exists()
    (target/'a'/'esm.npy').write_bytes(b'two')  # Same size and newer: treated as already copied.
    launcher.mirror(source, target)
    assert (target/'a'/'esm.npy').read_bytes() == b'two'
    (source/'a'/'esm.npy').write_bytes(b'three')
    (source/'b.json').write_text('{}')
    launcher.mirror(source, target)
    assert (target/'a'/'esm.npy').read_bytes() == b'three' and (target/'b.json').exists()


def test_parallel_dispatch_resumable_runs_and_failure_isolation(tmp_path, capsys):
    launcher = importlib.import_module('scripts.dgx_v4_validation')
    corpus, resources, cache, splits, config_path = fixture(tmp_path)
    # run_experiments refuses synthetic caches; this throwaway copy exercises the real path.
    manifest = json.loads((cache/'manifest.json').read_text())
    save_json(cache/'manifest.json', {k: v for k, v in manifest.items() if k != 'synthetic'})
    expand_holdouts(corpus, splits, tmp_path/'multi.json', 2, 1)
    config = json.loads(config_path.read_text())
    config.update(epochs=1, min_validation_genes=2)
    save_json(config_path, config)
    runs = tmp_path/'runs'

    def command(job):
        model, seed, fold = job
        return [sys.executable, ROOT/'scripts/run_experiments.py', '--corpus', corpus, '--resources', resources,
                '--cache', cache, '--splits', tmp_path/'multi.json', '--config', config_path, '--out', runs,
                '--folds', fold, '--seeds', seed, '--ablations', model, '--workers', 0]
    jobs = [('mipo', 42, 0), ('esm_mlp', 42, 0), ('mipo', 42, 99)]
    results = launcher.dispatch(jobs, ['', ''], command, lambda job: launcher.run_folder(runs, *job)/'train.log',
                                lambda slot: launcher.child_env(slot, 1), poll=.2)
    assert results[('mipo', 42, 0)] == 0 and results[('esm_mlp', 42, 0)] == 0 and results[('mipo', 42, 99)] != 0
    assert 'IndexError' in capsys.readouterr().out  # The failed run's error is shown, not only its log path.
    status = launcher.run_table(runs, ['mipo', 'esm_mlp'], [42], [0])
    assert status.complete.all() and set(status.stop_reason) <= {'epoch_cap', 'early_stopping'}
    again = launcher.dispatch(jobs[:1], [''], command, lambda job: launcher.run_folder(runs, *job)/'train.log',
                              lambda slot: launcher.child_env(slot, 1), poll=.2)
    assert again == {('mipo', 42, 0): 0}
    assert 'Completed:' in (launcher.run_folder(runs, 'mipo', 42, 0)/'train.log').read_text()
    tests = launcher.collect(status, 'test_metrics.csv')
    assert set(tests.model) == {'mipo', 'esm_mlp'} and set(tests.fold) == {0}
    assert set(status.torch) == {torch.__version__}
    provenance = launcher.run_folder(runs, 'esm_mlp', 42, 0)/'provenance.json'
    save_json(provenance, {**json.loads(provenance.read_text()), 'torch': 'other+build'})
    capsys.readouterr()
    launcher.report(types.SimpleNamespace(models=['mipo', 'esm_mlp'], seeds=[42], no_baseline=True),
                    runs, tmp_path, None, [0], None)
    assert 'different PyTorch builds' in capsys.readouterr().out  # e.g. Colab runs copied next to DGX runs


def test_gpu_choice_rejects_unusable_busy_and_explicit_small_gpus():
    launcher = importlib.import_module('scripts.dgx_v4_validation')
    gb = 1 << 30
    probes = {'0': {'ok': True, 'total': 80*gb, 'free': 79.5*gb}, '1': {'ok': True, 'total': 80*gb, 'free': 40*gb},
              '2': {'ok': True, 'total': 4*gb, 'free': 4*gb}, '9': {'ok': False}}
    choose = launcher.choose_gpus
    assert choose(['0', '2'], probes, explicit=False, allow_busy=False) == ['0']  # display GPU skipped
    for gpus, explicit in [(['0', '9'], True), (['0', '2'], True), (['0', '1'], False)]:
        with pytest.raises(SystemExit):
            choose(gpus, probes, explicit, allow_busy=False)
    assert choose(['0', '1'], probes, explicit=True, allow_busy=True) == ['0', '1']


def test_dispatch_retires_a_dead_gpu_instead_of_failing_the_queue(tmp_path):
    launcher = importlib.import_module('scripts.dgx_v4_validation')
    code = "import os, sys; sys.exit(3 if os.environ['CUDA_VISIBLE_DEVICES'] == 'bad' else 0)"
    jobs = [('mipo', 42, fold) for fold in range(3)]
    run = lambda slots: launcher.dispatch(  # noqa: E731
        jobs, slots, lambda job: [sys.executable, '-c', code], lambda job: tmp_path/f'{job[2]}.log',
        lambda slot: {**os.environ, 'CUDA_VISIBLE_DEVICES': slot}, poll=.05, healthy=lambda slot: slot != 'bad')
    results = run(['bad', 'good'])
    assert sorted(results.values()) == [0, 0, 3]
    results = run(['bad'])  # No slot left: remaining jobs are reported as never started, not waited on forever.
    assert results[jobs[0]] == 3 and results[jobs[1]] is None and results[jobs[2]] is None


@pytest.mark.skipif(torch.cuda.is_available(), reason='checks the CPU refusal')
def test_runs_refuse_to_fall_back_to_cpu(tmp_path):
    args = [sys.executable, str(ROOT/'scripts/run_experiments.py'), '--require-cuda']
    args += [x for key in ['corpus', 'resources', 'cache', 'splits', 'config', 'out'] for x in ['--'+key, str(tmp_path)]]
    result = subprocess.run(args, capture_output=True, text=True, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT)})
    assert result.returncode != 0 and 'refusing to train on CPU' in result.stderr


def test_reuse_takes_finished_runs_only_from_experiments_differing_in_seeds_or_folds(tmp_path):
    launcher = importlib.import_module('scripts.dgx_v4_validation')
    protocol = {'code_and_resources_sha256': 'c', 'config': {'seed': 42, 'lr': 1}, 'seeds': [42, 123], 'folds': [83]}

    def experiment(name, torch_version='T', **changes):
        folder = tmp_path/name
        save_json(folder/'protocol.json', {**protocol, **changes})
        run = launcher.run_folder(folder/'runs', 'mipo', 42, 83)
        for file in ['test_summary.json', 'training_status.json']:
            save_json(run/file, {})
        save_json(run/'provenance.json', {'torch': torch_version})
        return folder
    same = experiment('v4_validation_direct_650M_a', seeds=[42], folds=[83, 90])
    experiment('v4_validation_direct_650M_b', config={'seed': 42, 'lr': 2})  # Different settings.
    experiment('v4_validation_curriculum_650M_c')  # Different scope.
    new = tmp_path/'v4_validation_direct_650M_new'
    jobs = [('mipo', 42, 83), ('mipo', 123, 83)]
    assert launcher.reusable_runs(tmp_path, new, protocol, jobs, 'T') == \
        {('mipo', 42, 83): launcher.run_folder(same/'runs', 'mipo', 42, 83)}
    assert launcher.reusable_runs(tmp_path, new, protocol, jobs, 'other build') == {}
    found = launcher.reusable_runs(tmp_path, new, protocol, jobs, 'T')
    launcher.adopt_runs(found, new/'runs', copy_files=False)  # Plan only: nothing copied.
    assert not (new/'runs').exists()
    stale = launcher.run_folder(new/'runs', 'mipo', 42, 83).with_name('fold_83.copying')
    save_json(stale/'leftover.json', {})  # An earlier copy that was cut off.
    launcher.adopt_runs(found, new/'runs')
    copied = launcher.run_folder(new/'runs', 'mipo', 42, 83)
    assert (copied/'test_summary.json').exists() and not stale.exists() and not (copied/'leftover.json').exists()
    (copied/'test_summary.json').unlink()
    launcher.adopt_runs(found, new/'runs')  # A run already here, even unfinished, is never overwritten.
    assert not (copied/'test_summary.json').exists()
