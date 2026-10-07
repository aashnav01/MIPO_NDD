# 3. RESULTS: the four models and the zero-shot LLR, compared fairly.
# Run after cell 2 with TRAIN_NOW = False (its step 6 writes matched_llr_metrics.csv for every finished run).
# Main analysis: only seed/fold pairs where ALL four models finished on the original seed (matched).
# Second analysis: the three collapsed seed/fold pairs filled with replacement seeds, only if all four
# models finished on that replacement seed (otherwise it lists what is still missing).
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
try:
    from IPython.display import display
except ModuleNotFoundError:
    display = print

EXP = Path(os.environ.get('MIPO_EXP', '/content/drive/MyDrive/MIPO_NDD/v4_validation_mechanism_650M_bafe6e1c78fd218e'))
MODELS = ['mipo', 'esm_mlp', 'no_structure', 'no_global_kernel']
LLR = 'LLR (zero-shot)'
# Replacement seeds, fixed per collapsed seed/fold pair. Fold 2 uses 2123 because 1123 also collapsed there.
REPLACEMENT = {(42, 4): 1042, (123, 0): 1123, (123, 2): 2123}

protocol = json.loads((EXP/'protocol.json').read_text())
SEEDS, FOLDS = protocol['seeds'], protocol['folds']
split = json.loads((EXP/'splits.json').read_text())
GENE = {i: split['folds'][i]['test'][0] for i in FOLDS}
panel_path = (globals().get('PROJECT') or Path('.'))/'v4/robust/mechanism_panel.csv'
PART = pd.read_csv(panel_path).set_index('gene').part.to_dict() if panel_path.exists() else {}


def score(root, model, seed, fold):
    run = root/model/f'seed_{seed}'/f'fold_{fold:02d}'
    if not (run/'test_summary.json').exists():
        return 'missing', np.nan, np.nan, None
    metrics = run/'matched_llr_metrics.csv'
    if not metrics.exists():
        return 'no LLR file', np.nan, np.nan, None
    a = pd.read_csv(metrics)
    if a[['model_spearman', 'llr_spearman']].isna().any().any():
        return 'undefined Spearman', np.nan, np.nan, None
    history = run/'history.json'
    epochs = len(json.loads(history.read_text())) if history.exists() else None
    # One test gene per fold: the gene score is the mean over its assays.
    return 'finished', a.model_spearman.mean(), a.llr_spearman.mean(), epochs


def collect(root, cells):
    rows = []
    for model in MODELS:
        for seed, fold in cells:
            status, rho, llr, epochs = score(root, model, seed, fold)
            rows.append(dict(model=model, seed=seed, fold=fold, gene=GENE[fold], status=status,
                             spearman=rho, llr_spearman=llr, epochs=epochs))
    return pd.DataFrame(rows)


def matched_tables(runs, label):
    ok = runs[runs.status == 'finished']
    counts = ok.groupby(['seed', 'fold']).model.nunique()
    complete = counts[counts == len(MODELS)].reset_index()[['seed', 'fold']]
    matched = ok.merge(complete, on=['seed', 'fold'])
    wide = matched.pivot_table(index=['seed', 'fold', 'gene'], columns='model', values='spearman')[MODELS]
    llr_spread = matched.groupby(['seed', 'fold']).llr_spearman.agg(lambda x: x.max()-x.min()).max()
    assert llr_spread < 1e-9, f'LLR differs between models on the same test rows ({llr_spread})'
    wide[LLR] = matched.groupby(['seed', 'fold', 'gene']).llr_spearman.first()
    columns = MODELS+[LLR]
    print(f'\n=== {label}: {len(wide)} matched seed/fold pairs (all four models finished) ===')
    print('Mean test Spearman over matched pairs, and per seed:')
    per_seed = wide.groupby(level='seed')[columns].mean()
    summary = pd.concat([wide[columns].mean().rename('all matched pairs').to_frame().T, per_seed]).astype(object)
    summary.loc['mean ± SD across seeds'] = [f'{per_seed[c].mean():.3f} ± {per_seed[c].std():.3f}' for c in columns]
    display(summary.map(lambda v: round(v, 3) if isinstance(v, float) else v))
    print('Per test gene (mean over the matched seeds):')
    genes = wide.groupby(level='gene')[columns].mean()
    genes.insert(0, 'seeds', wide.groupby(level='gene').size())
    if PART:
        genes.insert(1, 'panel part', genes.index.map(PART))
    display(genes.round(3))
    print('Paired comparisons (gene level: each gene averaged over its matched seeds; 10 genes is a small sample):')
    pairs = [('mipo', 'esm_mlp'), ('mipo', 'no_structure'), ('mipo', 'no_global_kernel'),
             ('mipo', LLR), ('esm_mlp', LLR), ('no_structure', LLR), ('no_global_kernel', LLR)]
    rows = []
    for a, b in pairs:
        d = (genes[a]-genes[b]).dropna()
        cell = (wide[a]-wide[b]).dropna()
        p = wilcoxon(d).pvalue if len(d) >= 5 and (d != 0).any() else np.nan
        rows.append({'comparison': f'{a} − {b}', 'mean difference': round(d.mean(), 3),
                     'genes where first is higher': f'{(d > 0).sum()} of {len(d)}',
                     'seed/fold pairs where first is higher': f'{(cell > 0).sum()} of {len(cell)}',
                     'Wilcoxon p (genes)': round(p, 3) if np.isfinite(p) else 'n/a'})
    display(pd.DataFrame(rows))
    return wide


# Original seeds
runs = collect(EXP/'runs', [(s, f) for s in SEEDS for f in FOLDS])
if (runs.status == 'no LLR file').any():
    raise RuntimeError('Some finished runs have no matched_llr_metrics.csv. Run cell 2 with TRAIN_NOW = False first.')
print('Runs without a usable result on the original seed (count these as training collapses in the write-up):')
bad = runs[runs.status != 'finished']
display(pd.DataFrame([{'model': m, 'count': int((bad.model == m).sum()),
                       'seed/fold': ', '.join(f's{s}/f{f} ({GENE[f]})' for s, f in zip(bad[bad.model == m].seed, bad[bad.model == m].fold)) or '—'}
                      for m in MODELS]))
main = matched_tables(runs, 'MAIN ANALYSIS (original seeds)')
main.to_csv(EXP/'results_main_matched.csv')

# Replacement seeds (second analysis)
RS = EXP/'runs_reseeded'
if RS.exists():
    needs_llr = [p for p in RS.glob('*/seed_*/fold_*/test_summary.json') if not (p.parent/'matched_llr_metrics.csv').exists()]
    if needs_llr and all(k in globals() for k in ['PROJECT', 'CORPUS', 'RESOURCES', 'LOCAL_CACHE']):
        print(f'Writing LLR comparisons for {len(needs_llr)} replacement run(s)...')
        subprocess.run([sys.executable, str(PROJECT/'scripts'/'compare_cached_llr.py'), '--corpus', str(CORPUS),
                        '--resources', str(RESOURCES), '--cache', str(LOCAL_CACHE), '--runs', str(RS)],
                       check=True, capture_output=True)
    rep = collect(RS, [(new, fold) for (old, fold), new in REPLACEMENT.items()])
    print('\nReplacement-seed runs (a real run shows several epochs):')
    display(rep[['model', 'seed', 'fold', 'gene', 'status', 'epochs', 'spearman']].round(3))
    todo = rep[rep.status != 'finished']
    if len(todo):
        print('Second analysis not computed yet. Still needed for a matched replacement analysis:',
              ', '.join(f'{m} seed {s} fold {f}' for m, s, f in zip(todo.model, todo.seed, todo.fold)))
    else:
        back = {(new, fold): old for (old, fold), new in REPLACEMENT.items()}
        swapped = rep.assign(seed=[back[(s, f)] for s, f in zip(rep.seed, rep.fold)])  # file under the original seed
        filled = pd.concat([runs[~runs.set_index(['seed', 'fold']).index.isin(list(REPLACEMENT))], swapped])
        second = matched_tables(filled, 'SECOND ANALYSIS (collapsed pairs filled with replacement seeds, all four models)')
        second.to_csv(EXP/'results_with_replacements.csv')
print('\nReport the main analysis as the result. The second analysis is a sensitivity check; state which seeds were replaced and why.')
