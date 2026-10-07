# 4. OPTIONAL: train the replacement-seed runs that a matched second analysis needs
# (all four models on each replaced seed/fold pair). Run after cell 2; TRAIN_NOW = False is enough.
# About 2-3 hours with the runs in parallel. Then run cell 3 again.
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

EXP = Path(os.environ.get('MIPO_EXP', '/content/drive/MyDrive/MIPO_NDD/v4_validation_mechanism_650M_bafe6e1c78fd218e'))
MODELS = ['mipo', 'esm_mlp', 'no_structure', 'no_global_kernel']
# Same fixed replacement seeds as cell 3. Fold 2 uses 2123 because 1123 also collapsed there.
REPLACEMENT = {(42, 4): 1042, (123, 0): 1123, (123, 2): 2123}
RS = EXP/'runs_reseeded'
LOGS = EXP/'logs_reseeded'
ASIDE = EXP/'set_aside_partial_runs_reseeded'
for name in ['PROJECT', 'CORPUS', 'RESOURCES', 'LOCAL_CACHE', 'SPLITS', 'CONFIG', 'METADATA']:
    assert name in globals(), 'Run cell 2 first (TRAIN_NOW = False is enough).'
LOGS.mkdir(parents=True, exist_ok=True)


def run_dir(job):
    return RS/job[0]/f'seed_{job[1]}'/f'fold_{job[2]:02d}'


def label(job):
    return f'{job[0]} seed {job[1]} fold {job[2]}'


def set_aside(job, why):
    target = ASIDE/job[0]/f'seed_{job[1]}'/f"fold_{job[2]:02d}_{time.strftime('%Y%m%d-%H%M%S')}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir(job)), str(target))
    print(f'{label(job)}: {why}; earlier attempt kept at {target.relative_to(EXP)}', flush=True)


jobs = [(m, new, fold) for (old, fold), new in REPLACEMENT.items() for m in MODELS
        if not (RS/m/f'seed_{new}'/f'fold_{fold:02d}'/'test_summary.json').exists()]
print(f'{len(jobs)} replacement run(s) to train:', '; '.join(label(j) for j in jobs) or 'none', flush=True)

# A partial run resumes only on the torch/Python that started it; otherwise it starts over (kept, not deleted).
runtime = subprocess.check_output([sys.executable, '-c', 'import platform, torch; '
                                   'print(torch.__version__, platform.python_version())'], text=True).split()
for job in jobs:
    last = run_dir(job)/'last.pt'
    if last.exists():
        import torch
        saved = torch.load(last, map_location='cpu', weights_only=True)['provenance']
        if [saved['torch'], saved['python']] != runtime:
            set_aside(job, f"partial run from torch {saved['torch']} / Python {saved['python']}")

threads = str(max(1, (os.cpu_count() or 1)//max(1, len(jobs))))
env = {**os.environ, 'OMP_NUM_THREADS': threads, 'MKL_NUM_THREADS': threads, 'PYTHONUNBUFFERED': '1'}
running, results, retried = {}, {}, set()


def start(job):
    log = open(LOGS/f"{job[0]}_seed{job[1]}_fold{job[2]:02d}.log", 'a')
    log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.flush()
    command = [sys.executable, PROJECT/'scripts'/'run_experiments.py', '--corpus', CORPUS, '--resources', RESOURCES,
               '--cache', LOCAL_CACHE, '--splits', SPLITS, '--config', CONFIG, '--metadata', METADATA,
               '--out', RS, '--folds', job[2], '--seeds', job[1], '--ablations', job[0]]
    running[job] = (subprocess.Popen(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, env=env), log)
    print(f"{time.strftime('%H:%M')} started {label(job)}", flush=True)


def last_epoch(job):
    path = LOGS/f"{job[0]}_seed{job[1]}_fold{job[2]:02d}.log"
    epochs = []
    for line in path.read_text(errors='replace').rsplit('\n=== ', 1)[-1].splitlines():
        if line.startswith('{"epoch"'):
            try:
                epochs.append(json.loads(line)['epoch'])
            except ValueError:
                continue
    return epochs[-1] if epochs else '-'


queue = list(jobs)
started, last_status = time.time(), time.time()
try:
    while queue or running:
        if queue:
            start(queue.pop(0))
            if queue:
                time.sleep(60)  # stagger the starts
                continue
        for job, (proc, log) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del running[job]
            tail = (LOGS/f"{job[0]}_seed{job[1]}_fold{job[2]:02d}.log").read_text(errors='replace').rsplit('\n=== ', 1)[-1]
            if code == 0:
                results[job] = 'finished'
            elif 'Validation Spearman undefined' in tail:
                results[job] = 'collapsed on its replacement seed too (report it; do not re-roll again)'
            elif ('Resume provenance mismatch' in tail or 'config changed' in tail) and job not in retried:
                retried.add(job)
                set_aside(job, 'inputs differ from an earlier attempt')
                queue.append(job)
                continue
            else:
                results[job] = f'failed (exit {code}): ' + (tail.strip().splitlines() or ['no output'])[-1][:160]
            print(f"{time.strftime('%H:%M')} {label(job)}: {results[job]}", flush=True)
        if running and time.time()-last_status >= 600:
            last_status = time.time()
            print(f"--- {time.strftime('%H:%M')}: " + '; '.join(f'{label(j)} epoch {last_epoch(j)}' for j in running), flush=True)
        if running:
            time.sleep(30)
finally:
    for proc, log in running.values():
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
print(f'\nRun time: {(time.time()-started)/60:.0f} min')
for job in jobs:
    print(f'  {label(job):34s} {results.get(job, "stopped before finishing")}')
print('Now run cell 3 again for the second analysis.')
