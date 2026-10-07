"""Generate the portable notebook using standard nbformat; no hidden local paths."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# MIPO-NDD — complete Colab workflow

Select **Runtime → Change runtime type → GPU**. Put `mipo_ndd_colab_package.zip` in `MyDrive/MIPO_NDD/`, then run cells in order.

Start with the `quick` profile (150M frozen ESM, no LLR). Use `full` for ESM650 + masked LLR in a separate cache/run. Masked LLR preparation can dominate runtime; feature extraction and epoch checkpoints are resumable. No pretrained MIPO weights are supplied.

The notebook keeps experimental measurements separate and holds out whole genes. See `ARCHITECTURE.md` and `DATA_AND_EXPERIMENTS.md` in the package. Current data cannot validate six independent biological mechanisms; unsupported predictions are flagged.""")
code("""from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import zipfile, os, subprocess, sys, json, shutil

DRIVE = Path('/content/drive/MyDrive/MIPO_NDD')
ARCHIVE = DRIVE / 'mipo_ndd_colab_package.zip'
PROJECT = Path('/content/mipo_ndd')
PROJECT.mkdir(exist_ok=True)
assert ARCHIVE.exists(), f'Place the provided archive at {ARCHIVE}'
with zipfile.ZipFile(ARCHIVE) as z:
    for item in z.infolist():
        target = (PROJECT / item.filename).resolve()
        if target != PROJECT and PROJECT not in target.parents:
            raise ValueError('Archive path escapes project directory')
    z.extractall(PROJECT)
os.chdir(PROJECT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.[test]'], check=True)
print('Installed:', PROJECT)""")
md("""## Verify installation and data

The smoke command uses explicit synthetic features only. It verifies execution, not accuracy. Run it once in a new output folder.""")
code("""def run(*args):
    subprocess.run([sys.executable, '-m', 'mipo', *map(str, args)], check=True)

subprocess.run([sys.executable, '-m', 'pytest', '-q'], check=True)
import torch
print('PyTorch:', torch.__version__, 'GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
CORPUS = PROJECT / 'ndd_training_corpus_v3.csv'
RESOURCES = PROJECT / 'prepared/resources'
SPLITS = PROJECT / 'prepared/splits/gene.json'
METADATA = PROJECT / 'prepared/audit/assay_metadata.csv'
run('audit', '--corpus', CORPUS, '--out', PROJECT / 'artifacts/audit')
run('baselines', '--corpus', CORPUS, '--out', PROJECT / 'artifacts/baselines')
import pandas as pd
display(pd.read_csv(PROJECT / 'prepared/audit/coverage.csv'))""")
md("""## Choose a compute profile and holdout

`quick` is the first feasibility experiment. `full` enables 650M and masked LLR for every measured position. Use the same profile for a baseline/model comparison. Default graph sampling is 256 residues; the full-protein config is available separately. GPU memory and total runtime vary by Colab hardware.""")
code("""PROFILE = 'quick'  # 'quick' or 'full'
TEST_GENE = 'PTEN'
SEED = 42
assert PROFILE in {'quick', 'full'}
assert torch.cuda.is_available(), 'Enable a GPU runtime before feature extraction.'
MODEL = 'facebook/esm2_t30_150M_UR50D' if PROFILE == 'quick' else 'facebook/esm2_t33_650M_UR50D'
CACHE = DRIVE / f'cache_{PROFILE}'
RUN_ROOT = DRIVE / f'runs_{PROFILE}'
config = json.loads((PROJECT / 'configs/colab.json').read_text())
config.update(seed=SEED, use_llr=PROFILE == 'full', require_llr=PROFILE == 'full')
# For an initial short execution check, reduce these in a NEW run directory:
# config.update(epochs=2, steps_per_epoch=50)
config_path = PROJECT / f'active_{PROFILE}.json'
config_path.write_text(json.dumps(config, indent=2))
split = json.loads(SPLITS.read_text())
FOLD = next(f['fold'] for f in split['folds'] if TEST_GENE in f['test'])
print('Split:', split['folds'][FOLD])
print('Profile:', PROFILE, 'Feature model:', MODEL)""")
md("""## Frozen feature cache

Validated sequences and structure arrays are bundled. BRCA2 uses the missing-structure path. The full profile computes masked LLRs consistently for all genes, rather than filling expansion rows with zero. A restarted job reuses completed files. Do not change the model/window settings in an existing cache.""")
code("""args = ['features', '--corpus', CORPUS, '--resources', RESOURCES, '--cache', CACHE, '--model', MODEL]
if PROFILE == 'quick':
    args.append('--no-llr')
run(*args)
# Use local SSD during training; Drive remains the persistent cache.
LOCAL_CACHE = PROJECT / f'cache_{PROFILE}'
shutil.copytree(CACHE, LOCAL_CACHE, dirs_exist_ok=True)
if PROFILE == 'full':
    run('baselines', '--corpus', CORPUS, '--resources', RESOURCES, '--cache', LOCAL_CACHE, '--out', RUN_ROOT / 'zero_shot')""")
md("""## Train and evaluate one untouched test gene

Training selects the checkpoint only with the validation gene. Calibration uses a different gene, and final test labels never determine scaling, checkpoint choice or intervals. The command resumes at completed epochs if a checkpoint already exists. Changing configuration requires a new run directory.""")
code("""OUT = RUN_ROOT / f'mipo_seed{SEED}_fold{FOLD:02d}'
args = ['train', '--corpus', CORPUS, '--resources', RESOURCES, '--cache', LOCAL_CACHE,
        '--splits', SPLITS, '--config', config_path, '--metadata', METADATA,
        '--out', OUT, '--fold', FOLD]
if (OUT / 'last.pt').exists():
    args.append('--resume')
run(*args)
display(pd.read_csv(OUT / 'test_metrics.csv'))
print((OUT / 'test_summary.json').read_text())
display(pd.read_csv(OUT / 'test_predictions.csv').head())""")
md("""## Export a held-out variant's residue field

This example uses a measured position so its masked LLR is already cached. The `.npz` reports reference residue positions and learned fields, not an experimentally proven mechanism. Sparse mode reports only sampled residues.""")
code("""data = pd.read_csv(CORPUS, low_memory=False)
query = data[data.gene == TEST_GENE].head(1)[['gene', 'wt', 'ref_pos', 'mut', 'assay_id']].copy()
task_map = {'curated_functional_consensus':'consensus', 'activity':'activity', 'stability':'stability', 'organismal_fitness':'fitness'}
query['task'] = task_map[data[data.gene == TEST_GENE].assay_type.iloc[0]]
query['assay_key'] = query.gene + '::' + query.assay_id
query.to_csv(PROJECT / 'queries.csv', index=False)
run('predict', '--variants', PROJECT / 'queries.csv', '--resources', RESOURCES, '--cache', LOCAL_CACHE,
    '--checkpoint', OUT / 'best.pt', '--metadata', METADATA, '--out', OUT / 'query', '--export-fields')
display(pd.read_csv(OUT / 'query/predictions.csv'))
import numpy as np
field = np.load(OUT / 'query/field_000000.npz')
field_table = pd.DataFrame({'position': field['reference_positions'], 'response_norm': np.linalg.norm(field['scalar_field'], axis=1)})
display(field_table.sort_values('response_norm', ascending=False).head(20))""")
md("""## Launch a controlled comparison

Run only after reviewing the first fold's runtime and memory. Keep one architecture's aggregate report separate from other architectures. This runner is sequential and resumes unfinished runs. Replace the single fold with `all` and the seed with `42,123,2026` for the complete experiment. That can span many Colab sessions.""")
code("""RUN_COMPARISON = False
if RUN_COMPARISON:
    subprocess.run([sys.executable, 'scripts/run_experiments.py', '--corpus', str(CORPUS),
                    '--resources', str(RESOURCES), '--cache', str(LOCAL_CACHE), '--splits', str(SPLITS),
                    '--config', str(config_path), '--metadata', str(METADATA), '--out', str(RUN_ROOT / 'experiments'),
                    '--folds', str(FOLD), '--seeds', str(SEED), '--ablations', 'esm_mlp,mipo'], check=True)""")
md("""## Optional next experiments

- Add actual ProteinGym generic/family assays using `import-proteingym`, rebuild references/splits, then use the curriculum config.
- Use explicit homology clusters for family holdout (`DATA_AND_EXPERIMENTS.md`).
- Import real conformer ensembles with `mipo import-structure`; use a new run directory because feature provenance changes.
- Prepare paired WT/mutant physical teachers with `scripts/prepare_field_teachers.py`.
- Combine three seeds of the same fold with `scripts/ensemble.py`.

All model checkpoints and predictions are already on Drive. Do not treat a successful synthetic smoke test, a plausible residue map, or a single-gene correlation as validation of novelty or clinical utility.""")

nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.10"},
               "colab": {"name": "MIPO_NDD_Colab.ipynb", "provenance": []}, "accelerator": "GPU"}
nbf.validate(nb)
for cell in cells:
    if cell.cell_type == "code":
        compile(cell.source, "notebook", "exec")
Path("notebooks").mkdir(exist_ok=True)
nbf.write(nb, "notebooks/MIPO_NDD_Colab.ipynb")
print('Wrote validated Colab notebook')
