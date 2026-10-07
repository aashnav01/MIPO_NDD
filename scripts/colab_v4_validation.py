# MIPO-NDD V4 multi-gene validation on Colab: paste this entire file into ONE cell, or use
# notebooks/MIPO_NDD_V4_Validation_Colab.ipynb. It installs the uploaded release and runs
# scripts/dgx_v4_validation.py, the same launcher used on DGX/Linux servers: identical splits,
# checkpoint selection, ablations, feature-cache checks, matched comparisons and resumable runs.
# Defaults run the complete plan: both scopes x 3 seeds x 4 models on the PTEN fold (A100 GPU).
STORAGE = 'drive'                      # drive: Google Drive (survives disconnects; needed for multi-session runs); local: /content
ARCHIVE_NAME = 'mipo_ndd_v4_colab_v2.zip'
SCOPES = ['direct', 'curriculum']      # Run in this order; each scope is its own experiment folder
MODEL_SIZE = '650M'
USE_LLR = True
TEST_GENES = ['PTEN']                   # Add genes only as a predefined evaluation plan.
SEEDS = [42, 123, 2026]                # Model seeds; mean and spread across seeds show training variability
SPLIT_SEED = 42                        # Same validation/calibration split across model seeds.
VALIDATION_GROUPS = 5
CALIBRATION_GROUPS = 3
STEPS_PER_EPOCH = 3000
EPOCHS = 20                            # Direct mode only; curriculum uses its stage budgets.
PATIENCE = 6
DO_FEATURES = None                     # None: build what is missing; True: run (resumable) extraction; False: reuse only
FEATURE_SCOPE = 'all'                  # all: every V4 protein once, serving both scopes; scope: this corpus only (faster first build)
DO_TRAIN = True
DO_BASELINE = True
DO_ABLATIONS = True
ABLATIONS = 'esm_mlp,fixed_heads,no_structure'  # Main mipo runs once, even if also listed here.
REUSE = True                           # Copy finished runs from earlier folders that differ only in seeds (e.g. your seed-42 runs)
WORKERS = None                         # DataLoader workers per run; None: from the CPU count
PLAN_ONLY = False                      # True: show splits, homology check, feature status and reused runs; no GPU work
REPORT_ONLY = False                    # True: rebuild the tables from finished runs; no training
PROGRESS_MINUTES = 2                   # How often each running model reports its epoch

from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile

assert STORAGE in {'local', 'drive'} and not (PLAN_ONLY and REPORT_ONLY)
assert SCOPES and set(SCOPES) <= {'direct', 'curriculum'} and len(set(SCOPES)) == len(SCOPES)
if STORAGE == 'drive':
    from google.colab import drive
    drive.mount('/content/drive')
    STORAGE_ROOT = Path('/content/drive/MyDrive/MIPO_NDD')
else:
    STORAGE_ROOT = Path('/content')
ARCHIVE = STORAGE_ROOT/ARCHIVE_NAME
assert ARCHIVE.exists(), f'Upload {ARCHIVE_NAME} to {STORAGE_ROOT}'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


print('Installing', ARCHIVE, flush=True)
archive_hash = sha(ARCHIVE)
PROJECT = Path('/content')/f'mipo_ndd_v4_{archive_hash[:12]}'
PROJECT.mkdir(exist_ok=True)
PROJECT = PROJECT.resolve()
if not (PROJECT/'.extracted').exists():  # The marker is written last, so a cut-off extraction repeats.
    with zipfile.ZipFile(ARCHIVE) as archive:
        for item in archive.infolist():
            target = (PROJECT/item.filename).resolve()
            if target != PROJECT and PROJECT not in target.parents:
                raise ValueError('Unsafe archive path')
        archive.extractall(PROJECT)
    (PROJECT/'.extracted').write_text(archive_hash)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(PROJECT)], check=True)
if WORKERS is None and Path('/dev/shm').exists() and shutil.disk_usage('/dev/shm').total < 1 << 30:
    WORKERS = 0
    print('/dev/shm is under 1 GB, so data loading stays in the main process.')

FEATURE_TAG = f'v4_{MODEL_SIZE}_{"masked" if USE_LLR else "no_llr"}'
FEATURE_ARCHIVE = STORAGE_ROOT/f'{FEATURE_TAG}.zip'


def launcher_args(scope):
    args = ['--root', STORAGE_ROOT, '--scope', scope, '--model-size', MODEL_SIZE,
            '--test-genes', ','.join(TEST_GENES), '--seeds', ','.join(map(str, SEEDS)),
            '--split-seed', SPLIT_SEED, '--validation-groups', VALIDATION_GROUPS,
            '--calibration-groups', CALIBRATION_GROUPS, '--steps-per-epoch', STEPS_PER_EPOCH,
            '--epochs', EPOCHS, '--patience', PATIENCE,
            '--features', {None: 'auto', True: 'build', False: 'reuse'}[DO_FEATURES],
            '--feature-scope', FEATURE_SCOPE, '--cache', STORAGE_ROOT/FEATURE_TAG,
            '--ablations', ABLATIONS, '--progress-minutes', PROGRESS_MINUTES]
    flags = {'--no-llr': not USE_LLR, '--no-train': not DO_TRAIN, '--no-ablations': not DO_ABLATIONS,
             '--no-baseline': not DO_BASELINE, '--no-reuse': not REUSE,
             '--plan-only': PLAN_ONLY, '--report-only': REPORT_ONLY}
    args += [flag for flag, enabled in flags.items() if enabled]
    if WORKERS is not None:
        args += ['--workers', WORKERS]
    if STORAGE == 'drive':
        args += ['--local-cache', Path('/content')/FEATURE_TAG]  # Drive is slow for the random reads of training.
    if FEATURE_ARCHIVE.exists():
        args += ['--feature-archive', FEATURE_ARCHIVE]  # Unpacked only when the cache folder has no manifest yet.
    return args


EXPERIMENTS, FAILED = {}, []
for SCOPE in SCOPES:
    print(f'\n{"#"*78}\n# SCOPE: {SCOPE}\n{"#"*78}', flush=True)
    gpu_refused = False
    process = subprocess.Popen([sys.executable, '-u', str(PROJECT/'scripts/dgx_v4_validation.py'),
                                *map(str, launcher_args(SCOPE))],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=PROJECT,
                               env={**os.environ, 'TQDM_MININTERVAL': '30'})
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
            if line.startswith('Outputs: '):
                EXPERIMENTS[SCOPE] = Path(line[len('Outputs: '):].strip())
            gpu_refused = gpu_refused or line.startswith(('GPU memory already in use', 'CUDA cannot use GPU'))
        code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        print('\nStopped. Rerun this cell to resume: finished runs are kept, interrupted ones continue.')
        raise
    if code and gpu_refused:
        print('On Colab: Runtime > Restart session frees GPU memory held by earlier cells or runs; then rerun this cell.')
        raise RuntimeError('GPU check refused to start.')
    if code:
        FAILED.append(SCOPE)  # Other scopes still run; rerunning the cell resumes this one.
EXPERIMENT = EXPERIMENTS.get(SCOPES[-1])  # Kept for older download cells.

if not PLAN_ONLY:
    import pandas as pd
    tables = []
    for scope, folder in EXPERIMENTS.items():
        path = folder/'macro_comparison_by_seed.csv'
        if path.exists():
            tables.append(pd.read_csv(path).assign(scope=scope))
    if tables:
        runs = pd.concat(tables, ignore_index=True)
        summary = runs.groupby(['scope', 'model'], sort=False).agg(
            seeds=('seed', 'nunique'), spearman_mean=('model_spearman', 'mean'), spearman_sd=('model_spearman', 'std'),
            esm_llr=('llr_spearman', 'mean'), delta_mean=('delta', 'mean'), delta_sd=('delta', 'std'),
            seeds_beating_esm=('delta', lambda d: int((d > 0).sum())))
        print('\n'+'='*78+'\nSUMMARY: test macro Spearman vs the ESM2 LLR on the same rows, across seeds\n'+'='*78)
        print(summary.round(4).to_string())
        summary.to_csv(STORAGE_ROOT/'summary_all_scopes.csv')
        print('Saved', STORAGE_ROOT/'summary_all_scopes.csv')
        print('SD across seeds is training variability on these test genes, not uncertainty across genes.')
        print('Checkpoints were chosen on validation only. Do not pick a model or scope from these test numbers.')
if FAILED:
    raise RuntimeError(f'Some runs did not finish in scope(s) {FAILED}; read the messages above. Rerunning this cell resumes.')
if STORAGE == 'local' and not PLAN_ONLY:
    print('Runtime storage is temporary: run the download cell(s) before this runtime is deleted.')
