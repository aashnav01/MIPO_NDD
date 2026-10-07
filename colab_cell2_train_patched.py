# 2. TRAIN: resumes the DGX run in MyDrive/MIPO_NDD. Re-run this cell after any disconnect.
STORAGE = 'drive'                      # Runs live on Drive so a disconnect loses at most one epoch
SCOPE = 'mechanism'                    # mechanism (10 genes), direct, or curriculum
MODEL_SIZE = '650M'
USE_LLR = True
TEST_GENES = None                      # None: fixed 10-gene mechanism panel; direct/curriculum: 3-gene pilot
SEEDS = [42, 123, 2026]                # Training variability; seeds are not independent biological samples
SPLIT_SEED = 42                        # Same validation/calibration split across model seeds.
VALIDATION_GROUPS = 5
CALIBRATION_GROUPS = 3
STEPS_PER_EPOCH = 3000
EPOCHS = 20                            # Direct mode only; curriculum uses its stage budgets.
PATIENCE = 6
DO_FEATURES = False                    # Restored cache is complete; False never reloads ESM or rewrites the cache
DO_TRAIN = True
DO_BASELINE = True
DO_ABLATIONS = True
ABLATIONS = 'esm_mlp,no_structure,no_global_kernel'  # Matched component controls; one seed is a pilot only
EXPECTED_EXPERIMENT = 'v4_validation_mechanism_650M_bafe6e1c78fd218e'  # DGX run; None allows a new protocol folder
SEED_FIRST = True                      # Finish all 4 models for seed 42 before seed 123 (order only; same results)
PARALLEL = 10                          # Training processes sharing the GPU; 1 = one at a time. Same results either way
TRAIN_NOW = False                      # False: skip training, only write results (steps 5-7). True: train missing runs
SHOW_RUN_TABLES = False                # True prints every run's epoch tables in step 5 (very long with 100+ runs)

from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
try:
    import pandas as pd
except ModuleNotFoundError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pandas'], check=True)
    import pandas as pd
from IPython.display import display

# Colab provides /content; on a plain JupyterLab host (e.g. DGX) use the notebook's directory.
WORK_ROOT = Path('/content') if Path('/content').is_dir() else Path.cwd()

assert STORAGE in {'local', 'drive'}
if STORAGE == 'drive':
    from google.colab import drive
    drive.mount('/content/drive')
    STORAGE_ROOT = Path('/content/drive/MyDrive/MIPO_NDD')
else:
    STORAGE_ROOT = WORK_ROOT
ARCHIVE = STORAGE_ROOT/'mipo_ndd_v4_colab_upload.zip'
assert ARCHIVE.exists(), f'Upload the updated release ZIP to {ARCHIVE}'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def banner(text):
    print('\n'+'='*78+'\n'+text+'\n'+'='*78, flush=True)

def run_script(script, *args):
    subprocess.run([sys.executable, str(PROJECT/'scripts'/script), *map(str, args)], check=True)

def run(*args):
    subprocess.run([sys.executable, '-m', 'mipo', *map(str, args)], check=True)

banner('1. INSTALL UPDATED RELEASE')
archive_hash = sha(ARCHIVE)
PROJECT = WORK_ROOT/f'mipo_ndd_v4_validation_{archive_hash[:12]}'
PROJECT.mkdir(exist_ok=True)
with zipfile.ZipFile(ARCHIVE) as archive:
    for item in archive.infolist():
        target = (PROJECT/item.filename).resolve()
        if target != PROJECT and PROJECT not in target.parents:
            raise ValueError('Unsafe archive path')
    archive.extractall(PROJECT)
os.chdir(PROJECT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.[test]'], check=True)
import torch
assert torch.cuda.is_available(), 'Choose a GPU runtime'
print('GPU:', torch.cuda.get_device_name(0))
print('Release SHA256:', archive_hash)
assert SCOPE in {'mechanism', 'direct', 'curriculum'}
if TEST_GENES is None:
    TEST_GENES = (pd.read_csv(PROJECT/'v4/robust/mechanism_panel.csv').gene.tolist()
                  if SCOPE == 'mechanism' else ['ASPA', 'KCNQ2', 'PAX6'])
assert MODEL_SIZE in {'150M', '650M'}
assert SEEDS and TEST_GENES and len(set(SEEDS)) == len(SEEDS) and len(set(TEST_GENES)) == len(TEST_GENES)
assert VALIDATION_GROUPS >= 2 and CALIBRATION_GROUPS >= 1
assert STEPS_PER_EPOCH > 0 and EPOCHS > 0 and PATIENCE > 0

banner('2. FIX SPLITS AND CHECKPOINT SELECTION BEFORE TRAINING')
DATA_SCOPE = 'curriculum' if SCOPE == 'mechanism' else SCOPE
CORPUS = PROJECT/f'v4/robust/{DATA_SCOPE}.csv.gz'
RESOURCES = PROJECT/'v4/resources'
METADATA = PROJECT/'v4/robust/assay_metadata.csv'
SOURCE_SPLITS = PROJECT/f'v4/robust/{SCOPE}_splits.json'
print('Primary analysis excludes consensus and documented duplicate deposits; see v4/robust/build_report.json')
SPLITS = PROJECT/f'active_{SCOPE}_multigene_splits.json'
run_script('prepare_validation_splits.py', '--corpus', CORPUS, '--source', SOURCE_SPLITS,
           '--out', SPLITS, '--validation-groups', VALIDATION_GROUPS,
           '--calibration-groups', CALIBRATION_GROUPS, '--seed', SPLIT_SEED,
           '--report-test-genes', ','.join(TEST_GENES), '--resources', RESOURCES)
split = json.loads(SPLITS.read_text())
support = json.loads(SPLITS.with_suffix('.support.json').read_text())
identity = json.loads(SPLITS.with_suffix('.identity.json').read_text())
if identity['flagged']:
    print(f"WARNING: test genes within {identity['flag_threshold']:.0%} sequence identity of a train gene "
          f"(similarity diagnostic; assess homology-aware splits before interpreting generalization): {identity['flagged']}")
folds = sorted({i for gene in TEST_GENES for i, fold in enumerate(split['folds']) if gene in fold['test']})
found = {g for i in folds for g in split['folds'][i]['test']}
assert set(TEST_GENES) <= found, f'Test genes absent from outer folds: {set(TEST_GENES)-found}'
for i in folds:
    fold = split['folds'][i]
    print(f"Fold {i}: test={fold['test']}, validation={fold['validation']}, calibration={fold['calibration']}, train genes={len(fold['train'])}")
    counts = support[str(fold['fold'])]
    print('Number of genes per task in each role:')
    display(pd.DataFrame(counts).T)
    for task, roles in counts.items():
        if roles['test'] and (not roles['validation'] or not roles['calibration']):
            print(f'Task coverage gap: {task}. Validation/calibration transfer for this task is not established.')
# Joint training for the mixed NDD/non-NDD mechanism panel; NDD specialization would
# change the target population. The measured broad corpus remains gene-held-out.
CONFIG_SCOPE = 'direct' if SCOPE == 'mechanism' else SCOPE
config = json.loads((PROJECT/f'configs/v4_robust_{CONFIG_SCOPE}.json').read_text())
config.update(seed=SEEDS[0], use_llr=USE_LLR, require_llr=USE_LLR,
              steps_per_epoch=STEPS_PER_EPOCH, patience=PATIENCE,
              selection_metric='macro_gene_spearman', min_validation_genes=VALIDATION_GROUPS)
config['amp'] = False
if 'stages' in config:
    print('Curriculum stage budgets:', [(s['name'], s['epochs']) for s in config['stages']])
    max_epochs = sum(s['epochs'] for s in config['stages'])
else:
    config['epochs'] = EPOCHS
    max_epochs = EPOCHS
CONFIG = PROJECT/'active_v4_validation.json'
CONFIG.write_text(json.dumps(config, indent=2))
models = list(dict.fromkeys((['mipo'] if DO_TRAIN else []) +
                           ([x.strip() for x in ABLATIONS.split(',') if x.strip()] if DO_ABLATIONS else [])))
protocol = {'release_sha256': archive_hash, 'corpus_sha256': sha(CORPUS),
            'split_sha256': sha(SPLITS), 'metadata_sha256': sha(METADATA),
            'config': config, 'model_size': MODEL_SIZE, 'seeds': SEEDS, 'folds': folds, 'models': models, 'release_policy': 'robust_measured_v1'}
protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
EXPERIMENT = STORAGE_ROOT/f'v4_validation_{SCOPE}_{MODEL_SIZE}_{protocol_hash}'
if EXPECTED_EXPERIMENT and EXPERIMENT.name != EXPECTED_EXPERIMENT:
    raise RuntimeError(f'Protocol changed: this would train into {EXPERIMENT.name}, not {EXPECTED_EXPERIMENT}. '
                       'Check the release ZIP and the SEEDS/ABLATIONS/amp settings. Nothing was trained.')
EXPERIMENT.mkdir(parents=True, exist_ok=True)
for source, name in [(CONFIG, 'config.json'), (SPLITS, 'splits.json')]:
    shutil.copy2(source, EXPERIMENT/name)
(EXPERIMENT/'protocol.json').write_text(json.dumps(protocol, indent=2))
RUNS = EXPERIMENT/'runs'
print('Outputs:', EXPERIMENT)
print(f'{len(models)*len(SEEDS)*len(folds)} requested model/seed/fold runs: {models}')
print(f"{STEPS_PER_EPOCH} batches x {config['batch_size']} = {STEPS_PER_EPOCH*config['batch_size']:,} sample draws/epoch; max {max_epochs} epochs; patience {PATIENCE}")
print('Selection: mean assay Spearman within each validation gene, then equal mean across genes.')
print('Expanded holdouts change training data; compare models within this new protocol.')

banner('3. REUSE OR EXTRACT FROZEN FEATURES')
MODEL = {'150M': 'facebook/esm2_t30_150M_UR50D', '650M': 'facebook/esm2_t33_650M_UR50D'}[MODEL_SIZE]
FEATURE_TAG = f'v4_{MODEL_SIZE}_{"masked" if USE_LLR else "no_llr"}'
CACHE = STORAGE_ROOT/FEATURE_TAG
LOCAL_CACHE = CACHE if STORAGE == 'local' else PROJECT/FEATURE_TAG
FEATURE_ARCHIVE = STORAGE_ROOT/f'{FEATURE_TAG}.zip'
if not (CACHE/'manifest.json').exists() and FEATURE_ARCHIVE.exists():
    with zipfile.ZipFile(FEATURE_ARCHIVE) as archive:
        names = set(archive.namelist())
        if f'{FEATURE_TAG}/manifest.json' in names:
            destination = STORAGE_ROOT
        elif 'manifest.json' in names:
            destination = CACHE
        else:
            raise ValueError(f'{FEATURE_ARCHIVE} must contain manifest.json at its root or inside {FEATURE_TAG}/')
        destination.mkdir(parents=True, exist_ok=True)
        destination = destination.resolve()
        for item in archive.infolist():
            target = (destination/item.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError('Unsafe feature archive path')
        archive.extractall(destination)
    print('Uploaded feature cache extracted:', CACHE)
# Reuse embeddings but fill newly measured sites when an existing cache is supplied.
extract_features = DO_FEATURES is not False
if extract_features:
    print('Building/resuming ESM embeddings and masked LLR. This is a separate GPU stage before training.')
    args = ['features', '--corpus', CORPUS, '--resources', RESOURCES, '--cache', CACHE, '--model', MODEL]
    if not USE_LLR:
        args.append('--no-llr')
    run(*args)
assert (CACHE/'manifest.json').exists(), f'Feature cache missing: {CACHE}; set DO_FEATURES=True'
if CACHE.resolve() != LOCAL_CACHE.resolve():
    shutil.copytree(CACHE, LOCAL_CACHE, dirs_exist_ok=True)
print('Local cache ready:', LOCAL_CACHE)

banner('4. TRAIN AND RUN MATCHED ABLATIONS')
if models and not TRAIN_NOW:
    print('TRAIN_NOW = False: training skipped; results are computed from the runs already on Drive.')
if models and TRAIN_NOW:
    # train.py resumes a partial run only on the torch/Python that started it. A partial run from another
    # environment (e.g. the DGX's fold 7) is moved out of runs/, kept, and that fold restarts here.
    runtime = subprocess.check_output([sys.executable, '-c', 'import platform, torch; '
                                       'print(torch.__version__, platform.python_version())'], text=True).split()
    for last in sorted(RUNS.glob('*/seed_*/fold_*/last.pt')):
        partial = last.parent
        if (partial/'test_summary.json').exists():
            continue
        saved = torch.load(last, map_location='cpu', weights_only=True)['provenance']
        if [saved['torch'], saved['python']] != runtime:
            target = (EXPERIMENT/'set_aside_partial_runs'/partial.relative_to(RUNS).parent/
                      f"{partial.name}_{time.strftime('%Y%m%d-%H%M%S')}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(partial), str(target))
            print(f"Set aside partial run from torch {saved['torch']} / Python {saved['python']}; it restarts on "
                  f"torch {runtime[0]} / Python {runtime[1]}. Kept at {target.relative_to(EXPERIMENT)}", flush=True)

    # One process per model/seed/fold; each writes only its own run folder and its log in EXPERIMENT/logs.
    # A job starts when a slot is free, the previous start was STAGGER_S ago, and free RAM is at least
    # 1.5x the largest job measured so far plus 2 GB. One job always runs, whatever the RAM.
    # A run whose predictions collapse to a constant at epoch 0 is retried once, after all other runs.
    import platform
    import psutil
    POLL_S, STAGGER_S, STATUS_S, STALE_S = 20, 120, 600, 600
    order = ([(m, s, f) for s in SEEDS for f in folds for m in models] if SEED_FIRST else
             [(m, s, f) for m in models for s in SEEDS for f in folds])

    def run_dir(job):
        return RUNS/job[0]/f'seed_{job[1]}'/f'fold_{job[2]:02d}'

    def label(job):
        return f'{job[0]}_seed{job[1]}_fold{job[2]:02d}'

    pending = [job for job in order if not (run_dir(job)/'test_summary.json').exists()]
    print(f'{len(order)-len(pending)} of {len(order)} runs finished; {len(pending)} to train, up to {PARALLEL} at a time', flush=True)
    LOGS = EXPERIMENT/'logs'
    LOGS.mkdir(exist_ok=True)
    SESSION = EXPERIMENT/'.training_session.json'
    session = f'{platform.node()}-{os.getpid()}-{int(time.time())}'
    try:
        other = json.loads(SESSION.read_text())
    except (OSError, ValueError):
        other = None
    if other and time.time()-other['heartbeat'] < STALE_S:
        raise RuntimeError(f"Runtime {other['session']} was training in this folder {(time.time()-other['heartbeat'])/60:.0f} min ago. "
                           'Two runtimes would write the same runs. Stop the other one, wait 10 minutes, then run this cell again.')
    threads = str(max(1, (os.cpu_count() or 1)//PARALLEL))
    env = {**os.environ, 'OMP_NUM_THREADS': threads, 'MKL_NUM_THREADS': threads, 'PYTHONUNBUFFERED': '1'}
    running, peak_gb, kills, failed = {}, {}, {}, []
    collapses, deferred = {}, []
    slots, finished, last_launch, last_status = PARALLEL, 0, 0., 0.

    def rss_gb(proc):
        try:
            parent = psutil.Process(proc.pid)
            return sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)])/1024**3
        except psutil.Error:
            return 0.

    shown = {}

    def epoch_records(job):
        # A job may be mid-way through writing a line; skip incomplete lines instead of crashing.
        path = LOGS/f'{label(job)}.log'
        records = []
        for line in (path.read_text(errors='replace').splitlines() if path.exists() else []):
            if line.startswith('{"epoch"'):
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue
        return records

    def last_epoch(job):
        records = epoch_records(job)
        return records[-1]['epoch'] if records else '-'

    def print_new_epochs(job):
        records = epoch_records(job)
        for r in records[shown.get(job, len(records)):]:
            note = 'best so far' if r['improved'] else f"no improvement {r['bad_epochs']}/{PATIENCE}"
            print(f"{time.strftime('%H:%M')} {label(job)} epoch {r['epoch']}: loss {r['loss']:.3f}, "
                  f"val spearman {r['validation_spearman']:.3f} ({note})", flush=True)
        shown[job] = len(records)

    def gpu_use():
        try:
            return subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'],
                                  capture_output=True, text=True).stdout.strip()
        except OSError:
            return 'n/a'

    started = time.time()
    try:
        while pending or running or deferred:
            for job, (proc, log) in list(running.items()):
                peak_gb[job] = max(peak_gb.get(job, 0.), rss_gb(proc))
                print_new_epochs(job)

                code = proc.poll()
                if code is None:
                    continue
                log.close()
                del running[job]
                if code == 0:
                    finished += 1
                    print(f"{time.strftime('%H:%M')} finished {label(job)}", flush=True)
                elif code == -9 and kills.get(job, 0) < 2:
                    kills[job] = kills.get(job, 0)+1
                    slots = max(1, slots-1)
                    pending.insert(0, job)
                    print(f"{time.strftime('%H:%M')} {label(job)} was killed by the system, usually for lack of RAM. "
                          f"Requeued to resume from its last epoch; now at most {slots} at a time", flush=True)
                else:
                    text = (LOGS/(label(job)+'.log')).read_text(errors='replace').rsplit('\n=== ', 1)[-1]
                    if 'Validation Spearman undefined' in text and not collapses.get(job):
                        collapses[job] = 1
                        deferred.append(job)
                        print(f"{time.strftime('%H:%M')} {label(job)} collapsed to constant predictions at epoch 0; "
                              "retrying once after the other runs", flush=True)
                    else:
                        failed.append(job)
                        print(f"{time.strftime('%H:%M')} FAILED {label(job)} (exit {code}); see {LOGS/(label(job)+'.log')}", flush=True)
                        if len(failed) >= 3 and not finished and pending:
                            pending.clear()
                            print('Three runs failed before any finished; no new runs will start.', flush=True)
            if not pending and deferred and len(running) < slots:
                pending.extend(deferred)
                deferred.clear()
                print(f"{time.strftime('%H:%M')} retrying {len(pending)} collapsed run(s) once", flush=True)
            free = psutil.virtual_memory().available/1024**3
            need = 1.5*max(peak_gb.values(), default=0.)+2
            if pending and (not running or (len(running) < slots and time.time()-last_launch >= STAGGER_S and free >= need)):
                job = pending.pop(0)
                log = open(LOGS/f'{label(job)}.log', 'a')
                log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} session {session}\n")
                log.flush()
                command = [sys.executable, PROJECT/'scripts'/'run_experiments.py', '--corpus', CORPUS,
                           '--resources', RESOURCES, '--cache', LOCAL_CACHE, '--splits', SPLITS, '--config', CONFIG,
                           '--metadata', METADATA, '--out', RUNS, '--folds', job[2], '--seeds', job[1], '--ablations', job[0]]
                running[job] = (subprocess.Popen(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, env=env), log)
                shown[job] = len(epoch_records(job))
                last_launch = time.time()
                print(f"{time.strftime('%H:%M')} started {label(job)} ({len(running)} running, {free:.0f} GB RAM free)", flush=True)
            beat = SESSION.with_suffix('.tmp')
            beat.write_text(json.dumps({'session': session, 'heartbeat': time.time(), 'running': [label(job) for job in running]}))
            beat.replace(SESSION)
            if time.time()-last_status >= STATUS_S:
                last_status = time.time()
                done = sum((run_dir(job)/'test_summary.json').exists() for job in order)
                print(f"--- {time.strftime('%H:%M')}: {done}/{len(order)} runs finished, {len(pending)} waiting, "
                      f"{len(deferred)} collapsed and waiting for a retry, "
                      f"{free:.0f} GB RAM free (next start needs {need:.0f}), GPU {gpu_use()}", flush=True)
                for job, (proc, _) in running.items():
                    print(f'    {label(job):34s} last finished epoch {last_epoch(job)}, RAM {rss_gb(proc):.1f} GB', flush=True)
            time.sleep(POLL_S)
    finally:
        if running:
            print('Stopping running jobs; each resumes from its last finished epoch next time.', flush=True)
        for proc, _ in running.values():
            proc.terminate()
        for proc, log in running.values():
            try:
                proc.wait(60)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()
        try:
            if json.loads(SESSION.read_text())['session'] == session:
                SESSION.unlink()
        except (OSError, ValueError):
            pass
    print(f'Run time: {(time.time()-started)/60:.1f} min')
    if collapses:
        recovered = [label(job) for job in collapses if job not in failed]
        print(f'Collapsed at epoch 0 in this session: {[label(job) for job in collapses]}. '
              f'Recovered on retry: {recovered or "none"}.')
    if failed:
        raise RuntimeError(f'{len(failed)} runs failed: {[label(job) for job in failed]}. Their logs are in {LOGS}. '
                           'Run this cell again to retry them; each resumes from its last finished epoch.')

banner('5. TRAINING HISTORY AND VALIDATION DIAGNOSTICS')
selected_runs = []
for path in sorted(RUNS.glob('*/seed_*/fold_*/history.json')):
    if int(path.parent.name.split('_')[1]) not in folds or int(path.parent.parent.name.split('_')[1]) not in SEEDS:
        continue
    selected_runs.append(path.parent)
    if not SHOW_RUN_TABLES:
        continue
    print('\n', path.parent.relative_to(RUNS))
    history = pd.DataFrame(json.loads(path.read_text()))
    losses = pd.json_normalize(history.loss_components).add_prefix('train_')
    display(pd.concat([history[['epoch', 'loss', 'validation_spearman', 'lr', 'bad_epochs']], losses], axis=1).round(5))
    status = json.loads((path.parent/'training_status.json').read_text())
    print('Saved checkpoint / stopping:', status)
    per_gene = pd.DataFrame(history.validation_per_gene.tolist(), index=history.epoch)
    display(per_gene.round(4))
    print('Best checkpoint validation by gene and assay:')
    validation = pd.read_csv(path.parent/'validation_metrics.csv')
    display(validation[['gene', 'task', 'n', 'spearman', 'bias', 'mean_sigma', 'coverage_90_gaussian']].round(4))
    print('The history describes this validation panel; it does not establish the cause of a plateau.')

if DO_BASELINE and USE_LLR:
    banner('6. EXACT-ROW CACHED LLR BASELINE')
    run_script('compare_cached_llr.py', '--corpus', CORPUS, '--resources', RESOURCES,
               '--cache', LOCAL_CACHE, '--runs', RUNS)

banner('7. TEST RANKING AND INTERVAL DIAGNOSTICS')
if SCOPE == 'mechanism' and selected_runs:
    run_script('evaluate_property_contrasts.py', '--runs', RUNS,
               '--out', EXPERIMENT/'paired_property_contrasts.csv')
    print('Paired rank contrasts are descriptive assay-separation diagnostics, not causal folding labels.')
comparisons, intervals = [], []
for folder in selected_runs:
    model = folder.parents[1].name
    seed = int(folder.parent.name.split('_')[1])
    fold = int(folder.name.split('_')[1])
    metrics_path = folder/'test_metrics.csv'
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        metrics.insert(0, 'model', model)
        metrics.insert(1, 'seed', seed)
        metrics.insert(2, 'fold', fold)
        intervals.append(metrics)
    comparison = folder/'matched_llr_metrics.csv'
    if comparison.exists():
        rows = pd.read_csv(comparison)
        rows.insert(0, 'model', model)
        rows.insert(1, 'seed', seed)
        rows.insert(2, 'fold', fold)
        comparisons.append(rows)
if comparisons:
    table = pd.concat(comparisons, ignore_index=True)
    if SCOPE == 'mechanism':
        panel = pd.read_csv(PROJECT/'v4/robust/mechanism_panel.csv').set_index('gene')
        table['evaluation_part'] = table.gene.map(panel.part)
        table['previously_examined'] = table.gene.map(panel.previously_examined)
    else:
        table['evaluation_part'] = 'ndd_evaluation'
    table.to_csv(EXPERIMENT/'matched_comparison.csv', index=False)
    display(table.round(4))
    if table[['model_spearman', 'llr_spearman']].notna().all().all():
        gene_scores = table.groupby(['model', 'seed', 'evaluation_part', 'gene'])[['model_spearman', 'llr_spearman', 'delta']].mean()
        per_seed = gene_scores.groupby(['model', 'seed', 'evaluation_part']).mean()
        print('Macro across genes, separately for each seed and NDD/non-NDD panel (all assays retained):')
        display(per_seed.round(4))
        per_seed.to_csv(EXPERIMENT/'macro_comparison_by_seed.csv')
        if len(SEEDS) > 1:
            print('Mean and standard deviation across seeds; this is training variability, not uncertainty across genes:')
            display(per_seed.groupby(['model', 'evaluation_part']).agg(['mean', 'std']).round(4))
    else:
        print('Some correlations are undefined: inspect the assay table; headline macro comparison suppressed.')
if intervals:
    table = pd.concat(intervals, ignore_index=True)
    table.to_csv(EXPERIMENT/'test_diagnostics.csv', index=False)
    display(table[['model', 'seed', 'gene', 'task', 'n', 'spearman', 'bias', 'mean_sigma',
                   'coverage_90_gaussian', 'coverage_calibrated', 'calibrated_interval_rows']].round(4))
    print('Gaussian coverage uses raw predicted sigma. Calibrated coverage uses separate calibration groups.')
    print('Missing calibrated coverage means no supported calibration interval. Nominal coverage is 0.90; transfer to unseen genes is empirical.')
    print('Bias, sigma, RMSE and NLL are in training-task-transformed units. Raw-score predictions and intervals are also saved.')
print(f'Evaluated {len(folds)} outer folds and {len(SEEDS)} seeds. Test results must not select checkpoints or tune this protocol.')
banner(f'DONE: {EXPERIMENT}')
if STORAGE == 'local':
    print('Download results before this runtime is deleted. Use the notebook download cell, or zip EXPERIMENT yourself.')
