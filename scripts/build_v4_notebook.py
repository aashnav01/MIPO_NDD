"""Produce a V4 Colab notebook with explicit NDD-only and full-curriculum modes."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md('''# MIPO-NDD V4 — tiered NDD training

The release contains actual measurements, not manifest-only plans. Choose **direct** for the verified NDD subset or **curriculum** for generic → family-teacher → verified NDD training. All reference constructs from the same gene remain in the same split.

Upload `mipo_ndd_v4_complete.zip` to `MyDrive/MIPO_NDD/`. Select a Colab GPU runtime. See `v4/DATASET_CARD.md` for evidence rules, proxy labels and exclusions. Default extraction uses frozen ESM150 without LLR for a manageable first run; ESM650 is selectable.''')
code('''from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import sys, subprocess, zipfile, os, json, shutil
DRIVE = Path('/content/drive/MyDrive/MIPO_NDD')
ARCHIVE = DRIVE/'mipo_ndd_v4_complete.zip'
PROJECT = Path('/content/mipo_ndd_v4')
PROJECT.mkdir(exist_ok=True)
assert ARCHIVE.exists(), f'Upload the provided ZIP to {ARCHIVE}'
with zipfile.ZipFile(ARCHIVE) as z:
    for item in z.infolist():
        target = (PROJECT/item.filename).resolve()
        if target != PROJECT and PROJECT not in target.parents:
            raise ValueError('Unsafe archive path')
    z.extractall(PROJECT)
os.chdir(PROJECT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.[test]'], check=True)
def run(*args):
    subprocess.run([sys.executable, '-m', 'mipo', *map(str, args)], check=True)
''')
md('''## Inspect the release

Tier A preserves the original consensus data but no longer calls every legacy gene “NDD gold.” B and C require explicit NDD evidence. C includes both physical stability assays and clearly labeled domain-abundance stability proxies. D and E are support data, not direct NDD evidence.''')
code('''import pandas as pd
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
print((PROJECT/'v4/audit/summary.json').read_text())
display(pd.read_csv(PROJECT/'v4/data/gene_evidence.csv').head())
assays = pd.read_csv(PROJECT/'v4/data/assay_metadata.csv')
display(assays.groupby(['supervision_tier', 'task']).loaded_rows.sum().unstack(fill_value=0))
subprocess.run([sys.executable, '-m', 'pytest', '-q'], check=True)
''')
md('''## Choose the training scope

The first run should use `SCOPE='direct'`, `MODEL_SIZE='150M'`, and `USE_LLR=False`. Full curriculum includes nearly one million measurements and is a substantially larger experiment. The model still samples gene-balanced batches rather than reading every row per epoch.

The test/validation/calibration genes come only from the verified NDD set, even when generic teachers are loaded. Exact identical protein references across genes share a split group. This is not yet a full homology-family benchmark.''')
code('''SCOPE = 'direct'  # direct or curriculum
MODEL_SIZE = '150M'  # 150M or 650M
USE_LLR = False
TEST_GENE = 'PTEN'
SEED = 42
assert SCOPE in {'direct', 'curriculum'}
assert MODEL_SIZE in {'150M', '650M'}
assert torch.cuda.is_available(), 'Choose a GPU runtime before ESM extraction'
MODEL = {'150M':'facebook/esm2_t30_150M_UR50D', '650M':'facebook/esm2_t33_650M_UR50D'}[MODEL_SIZE]
CORPUS = PROJECT/'v4/data'/('ndd_direct_v4.csv.gz' if SCOPE == 'direct' else 'ndd_training_corpus_v4.csv.gz')
RESOURCES = PROJECT/'v4/resources'
SPLITS = PROJECT/f'v4/splits/{SCOPE}.json'
METADATA = PROJECT/'v4/data/assay_metadata.csv'
config = json.loads((PROJECT/f'configs/v4_{"direct" if SCOPE == "direct" else "curriculum"}.json').read_text())
config.update(seed=SEED, use_llr=USE_LLR, require_llr=USE_LLR)
CONFIG = PROJECT/'active_v4.json'
CONFIG.write_text(json.dumps(config, indent=2))
split = json.loads(SPLITS.read_text())
FOLD = next(f['fold'] for f in split['folds'] if TEST_GENE in f['test'])
print('Holdout:', split['folds'][FOLD])
FEATURE_TAG = f'v4_{MODEL_SIZE}_{"masked" if USE_LLR else "no_llr"}'
CACHE = DRIVE/FEATURE_TAG
OUT = DRIVE/f'v4_runs_{SCOPE}_{MODEL_SIZE}_{USE_LLR}'/f'mipo_seed{SEED}_fold{FOLD:03d}'
''')
md('''## Extract features once per exact protein reference

Position 20 of one domain is not position 20 of another construct or the full protein. V4 uses `protein_reference_id` for sequence/structure/LLR caching and `gene` for holdouts. Multiple assays sharing an exact reference reuse the same cache. Structures that cannot be mapped by exact sequence matching remain explicitly masked.''')
code('''args = ['features', '--corpus', CORPUS, '--resources', RESOURCES, '--cache', CACHE, '--model', MODEL]
if not USE_LLR:
    args.append('--no-llr')
run(*args)
LOCAL_CACHE = PROJECT/FEATURE_TAG
shutil.copytree(CACHE, LOCAL_CACHE, dirs_exist_ok=True)
''')
md('''## Train, resume and evaluate

Consensus rows potentially overlapping newly restored assay measurements carry a documented 0.25 loss weight. No score is averaged across assays. This is a conservative modeling choice, not proof that the original consensus sources have been reconstructed; run a no-consensus sensitivity analysis for publication.''')
code('''args = ['train', '--corpus', CORPUS, '--resources', RESOURCES, '--cache', LOCAL_CACHE,
        '--splits', SPLITS, '--config', CONFIG, '--out', OUT, '--fold', FOLD, '--metadata', METADATA]
if (OUT/'last.pt').exists():
    args.append('--resume')
run(*args)
print((OUT/'test_summary.json').read_text())
display(pd.read_csv(OUT/'test_metrics.csv'))
display(pd.read_csv(OUT/'test_predictions.csv').head())
''')
md('''## Predict using the exact assay construct

Take a measured variant for this first demonstration so its features are cached. A prediction in an unobserved measurement type remains exploratory; an isolated-domain result is not automatically a full-protein functional effect.''')
code('''from mipo.corpus import read_corpus
data = read_corpus(CORPUS)
query = data[data.gene == TEST_GENE].head(1)[['gene', 'protein_reference_id', 'wt', 'ref_pos', 'mut', 'task', 'assay_key']]
query.to_csv(PROJECT/'query_v4.csv', index=False)
run('predict', '--variants', PROJECT/'query_v4.csv', '--resources', RESOURCES, '--cache', LOCAL_CACHE,
    '--checkpoint', OUT/'best.pt', '--metadata', METADATA, '--out', OUT/'query', '--export-fields')
display(pd.read_csv(OUT/'query/predictions.csv'))
''')
md('''## Optional full benchmark

Use one fold first to measure GPU memory and runtime. Then run fixed-seed ablations, and finally all NDD folds and multiple seeds. Set `--folds all --seeds 42,123,2026` in the runner for the larger benchmark; this will require many sessions. Outputs and epoch checkpoints stay on Drive.''')
code('''RUN_ABLATIONS = False
if RUN_ABLATIONS:
    subprocess.run([sys.executable, 'scripts/run_experiments.py', '--corpus', str(CORPUS), '--resources', str(RESOURCES),
                    '--cache', str(LOCAL_CACHE), '--splits', str(SPLITS), '--config', str(CONFIG), '--metadata', str(METADATA),
                    '--out', str(OUT.parent/'ablations'), '--folds', str(FOLD), '--seeds', str(SEED),
                    '--ablations', 'esm_mlp,fixed_heads,no_structure,no_global_kernel,mipo'], check=True)
''')
nb.cells = cells
nb.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
               'language_info': {'name': 'python', 'version': '3.10'}, 'accelerator': 'GPU',
               'colab': {'name': 'MIPO_NDD_V4_Colab.ipynb', 'provenance': []}}
nbf.validate(nb)
for cell in nb.cells:
    if cell.cell_type == 'code':
        compile(cell.source, 'v4_notebook', 'exec')
nbf.write(nb, 'notebooks/MIPO_NDD_V4_Colab.ipynb')
print('V4 Colab notebook written and validated')
