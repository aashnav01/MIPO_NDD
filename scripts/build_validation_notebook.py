"""Build the Colab notebook from the maintained Python entry point."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
source = (root/'scripts/colab_v4_validation.py').read_text(encoding='utf-8')
compile(source, 'colab_v4_validation.py', 'exec')
notebook = nbf.v4.new_notebook(cells=[
    nbf.v4.new_markdown_cell('''# MIPO-NDD: complete multi-gene validation (Colab, A100)

**Plan (defaults):** PTEN test fold; 4 matched models (`mipo`, `esm_mlp`, `fixed_heads`,
`no_structure`); seeds 42, 123 and 2026; both scopes, `direct` then `curriculum`; 650M ESM2
with masked LLR. That is 24 runs. Checkpoints are chosen on validation only, and every model
is compared with the ESM2 LLR on exactly the same PTEN rows.

1. **Runtime > Change runtime type > A100 GPU** (any GPU works, only slower).
2. Put `mipo_ndd_v4_colab_v2.zip` in `MyDrive/MIPO_NDD/`, replacing any older copy.
   Keep `STORAGE = 'drive'`: the full plan takes more than one session, and Drive keeps
   features, checkpoints and results between sessions.
3. Run the cell once with `PLAN_ONLY = True`. It shows the folds, the homology check,
   the feature status and which finished runs will be reused. No GPU time.
4. Set `PLAN_ONLY = False` and run it. After a disconnect or timeout, just run it again:
   finished runs are skipped and interrupted ones continue from their last epoch.

**Reuse:** runs you already finished (for example the four seed-42 `direct` runs) are copied
into the new experiment folder rather than retrained, when their folder differs only in
the seed list. Every input hash is checked again before a copied run is accepted.

**Time on a Colab A100 (12 CPUs):** direct runs took about 3 minutes per epoch and 35-45 minutes
per run. The 8 new direct runs take about 5-6 hours. Curriculum is 12 runs of up to 16 epochs;
its speed has not been measured, roughly 8-10 hours if epochs take as long. Features are
already on Drive and are not rebuilt. To do less, set `SCOPES = ['direct']` or fewer `SEEDS`.

**At the end** the cell prints one summary: for each scope and model, the PTEN macro Spearman
and its difference from the ESM2 LLR, as mean and SD over seeds, and how many seeds beat
the LLR. It is saved as `MyDrive/MIPO_NDD/summary_all_scopes.csv`. Each experiment folder
also keeps its full tables (`matched_comparison.csv`, `test_diagnostics.csv`, `run_status.csv`).

**If a run fails**, the cell prints its last log lines and continues with the other runs and
scopes. `Resume provenance mismatch` means Colab updated PyTorch since that run was
interrupted: delete that one run folder (`.../runs/<model>/seed_<s>/fold_<ff>`) and rerun.
A refusal because GPU memory is in use: Runtime > Restart session, then rerun.
See `DGX_A100.md` for the same launcher on a DGX.
'''),
    nbf.v4.new_code_cell(source),
    nbf.v4.new_markdown_cell('''## Download results

Zips every experiment folder from the run above, plus the summary. Checkpoints (`.pt`, about
25 MB per run) are left out unless `INCLUDE_CHECKPOINTS = True`; with Drive storage they stay
on Drive anyway.
'''),
    nbf.v4.new_code_cell('''from google.colab import files
import zipfile
from pathlib import Path
INCLUDE_CHECKPOINTS = False
assert globals().get('EXPERIMENTS'), 'Run the training cell first (REPORT_ONLY = True is enough)'
result_zip = Path('/content') / 'mipo_ndd_v4_results.zip'
with zipfile.ZipFile(result_zip, 'w', zipfile.ZIP_DEFLATED) as z:
    for folder in EXPERIMENTS.values():
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.name != '.lock' and (INCLUDE_CHECKPOINTS or path.suffix != '.pt'):
                z.write(path, path.relative_to(folder.parent))
    summary = STORAGE_ROOT / 'summary_all_scopes.csv'
    if summary.exists():
        z.write(summary, summary.name)
print(result_zip, f'{result_zip.stat().st_size / 2**20:.0f} MB')
files.download(str(result_zip))
'''),
    nbf.v4.new_markdown_cell('''## Optional: keep the feature cache (local storage)

Building features takes GPU time. With `STORAGE = 'local'`, download the cache and upload
it next session as `v4_650M_masked.zip` (the name follows `MODEL_SIZE` and `USE_LLR`).
With Drive storage it is already saved.
'''),
    nbf.v4.new_code_cell('''from google.colab import files
import shutil
from pathlib import Path
if STORAGE == 'drive':
    print('The feature cache is already on Drive:', STORAGE_ROOT / FEATURE_TAG)
else:
    assert (STORAGE_ROOT / FEATURE_TAG / 'manifest.json').exists(), 'No feature cache yet'
    cache_zip = shutil.make_archive(str(Path('/content') / FEATURE_TAG), 'zip',
                                    root_dir=STORAGE_ROOT, base_dir=FEATURE_TAG)
    files.download(cache_zip)
''')])
notebook.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                     'language_info': {'name': 'python', 'version': '3.10'}, 'accelerator': 'GPU'}
nbf.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type == 'code':
        compile(cell.source, 'colab_cell', 'exec')
for name in ['MIPO_NDD_V4_Upload_Colab.ipynb', 'MIPO_NDD_V4_Validation_Colab.ipynb']:
    path = root/'notebooks'/name
    nbf.write(notebook, path)
    print(path)
