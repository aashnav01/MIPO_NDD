# Running the V4 multi-gene validation on a DGX A100

`scripts/dgx_v4_validation.py` replaces the one-cell Colab launcher
(`scripts/colab_v4_validation.py`). The protocol is the same: split expansion,
config, validation-only checkpoint selection, ablations and the exact-row LLR
comparison. The DGX version runs independent model/seed/fold jobs on separate
GPUs at the same time, loads data with worker processes, and checks the feature
cache before any training starts. Rerunning the same command resumes everything.

The same launcher has run end to end on a Colab A100 (12 CPUs): PTEN fold, seed 42,
4 models. Each epoch took about 3 minutes and each run 35-45 minutes, one run at a
time. It has not yet run on a DGX, so parallel runtime and memory there are unmeasured.

## 1. Copy the project to the DGX

Use `dist/mipo_ndd_v4_dgx.zip`. It is the V4 release plus these DGX changes. Put
it on the node's local NVMe (`/raid` on a DGX A100). Home directories are often
small NFS mounts.

```bash
# from your PC (Windows PowerShell or Linux/macOS; quote the path, it has spaces)
scp "dist/mipo_ndd_v4_dgx.zip" <user>@<dgx-host>:/raid/<user>/
# on the DGX (without unzip: python3 -m zipfile -e mipo_ndd_v4_dgx.zip mipo_ndd_v4)
cd /raid/$USER && unzip -q mipo_ndd_v4_dgx.zip -d mipo_ndd_v4 && cd mipo_ndd_v4
```

If `/raid` does not exist or is not writable for you, ask the admin which disk to
use for large scratch data, and use that path instead.

## 2. Create the environment (once)

Run this on a machine with internet access. A login node works; it does not need a GPU.

```bash
bash scripts/dgx/setup_env.sh
```

It creates `.venv/` and chooses PyTorch CUDA 12.1 wheels, or CUDA 11.8 wheels on
drivers older than R525. It installs the package, downloads the ESM2 650M and 150M
weights into `.hf_cache/`, then runs the test suite. You can override its choices:

| Variable | Meaning |
|---|---|
| `PYTHON=python3.11` | Python ≥3.10 required. DGX OS 5 ships 3.8. If the system Python is too old, or `venv` fails because `python3-venv` is missing and you have no admin rights: install Miniforge, `conda create -n mipo python=3.11`, activate it, then run with `PYTHON=python`. |
| `TORCH_INDEX=https://download.pytorch.org/whl/cu118` | Force a wheel set. Check the driver with `nvidia-smi`. |
| `MIPO_ENV=/path/venv`, `HF_HOME=/path/hf` | Other locations. GPU nodes must be able to see both. |

**Container alternative (NGC).** If your site runs jobs in containers, start one with
`--ipc=host`. Without it, DataLoader workers crash on Docker's 64 MB `/dev/shm`.

```bash
docker run --gpus '"device=0,1,2,3"' --ipc=host -it -v /raid/$USER:/raid/$USER \
  -w /raid/$USER/mipo_ndd_v4 nvcr.io/nvidia/pytorch:24.05-py3 bash
pip install --no-deps -e . && pip install "pandas>=2,<3" "pyarrow>=15" "scipy>=1.10" \
  "biopython>=1.83" "transformers>=4.44,<5" safetensors tqdm requests "pytest>=8,<9" "nbformat>=5,<6"
```

`--no-deps` keeps the container's CUDA-matched PyTorch. Then call
`python scripts/dgx_v4_validation.py ...` directly instead of the run script.

## 3. Optional: reuse your Colab feature cache

If you downloaded `v4_650M_masked.zip` from Colab, copy it over and add
`--feature-archive /raid/$USER/v4_650M_masked.zip`. The launcher unpacks it and
checks that it was built with the same ESM model and LLR mode. It then fills
anything missing. By default the cache covers **all** V4 proteins and sites
(598 proteins, 51,599 masked sites), so a later curriculum run never changes the
files that earlier runs recorded. Without an archive the launcher builds the cache
on the first GPU before training starts.

## 4. Dry run (no GPU needed)

```bash
source .venv/bin/activate
python scripts/dgx_v4_validation.py --root /raid/$USER/mipo_ndd --plan-only
```

This prints the PTEN fold's roles and task coverage. It also prints the homology
check, the experiment folder name, the feature-cache status and the job plan.
Expected output for the default settings: test `PTEN`; validation `DLG4, HUWE1, OTC,
YAP1, ZNF292`; calibration `HRAS, PSAT1, UBTF`; 126 training genes; 4 runs.

## 5. Run

**With Slurm** (submit from the project root; edit the `#SBATCH` lines for your
partition, GPU count and time limit):

```bash
sbatch scripts/dgx/run_v4_validation.sh --root /raid/$USER/mipo_ndd
```

**Without Slurm** (direct SSH to a DGX). First check which GPUs are free with
`nvidia-smi`, then run inside `tmux` so the job survives an SSH disconnect:

```bash
tmux new -s mipo
bash scripts/dgx/run_v4_validation.sh --root /raid/$USER/mipo_ndd --gpus 0,1,2,3
# detach: Ctrl-b d      reattach: tmux attach -t mipo
```

`--gpus` uses the numbers `nvidia-smi` shows (the launcher sets
`CUDA_DEVICE_ORDER=PCI_BUS_ID`). Before any GPU work it opens each chosen GPU the way
a run will. It stops if CUDA cannot use a GPU, or if another program already holds
more than 2 GB on it (`--allow-busy-gpus` overrides the second check). Without
`--gpus`, GPUs under 10 GB, such as a DGX Station's display GPU, are skipped. Each run
refuses to start on CPU, so a wrong GPU id fails at once instead of running for days.

The launcher builds features once, then starts one run per GPU. The default plan
has 4 runs: `mipo`, `esm_mlp`, `fixed_heads` and `no_structure`, with seed 42 on the
PTEN fold. Extra runs wait for a free GPU. If a run fails and its GPU no longer
responds, that GPU gets no more runs; the rest continue on the other GPUs.

### Colab setting → DGX flag

| Colab cell | DGX launcher |
|---|---|
| `STORAGE` | `--root DIR` (or `MIPO_ROOT`) |
| `SCOPES` | `--scope direct\|curriculum` (one launch per scope) |
| `MODEL_SIZE` | `--model-size 650M\|150M` |
| `USE_LLR = False` | `--no-llr` |
| `TEST_GENES` | `--test-genes PTEN,SCN2A` |
| `SEEDS` | `--seeds 42,123,2026` |
| `SPLIT_SEED`, `VALIDATION_GROUPS`, `CALIBRATION_GROUPS` | `--split-seed`, `--validation-groups`, `--calibration-groups` |
| `STEPS_PER_EPOCH`, `EPOCHS`, `PATIENCE` | `--steps-per-epoch`, `--epochs`, `--patience` |
| `DO_FEATURES` None / True / False | `--features auto` / `build` / `reuse` |
| `DO_TRAIN=False`, `DO_ABLATIONS=False`, `DO_BASELINE=False` | `--no-train`, `--no-ablations`, `--no-baseline` |
| `ABLATIONS` | `--ablations esm_mlp,fixed_heads,no_structure` |
| *(new)* | `--gpus`, `--jobs-per-gpu`, `--allow-busy-gpus`, `--no-reuse`, `--workers`, `--cache`, `--local-cache`, `--feature-archive`, `--feature-scope`, `--progress-minutes`, `--plan-only`, `--report-only` |

For Colab, the notebook `notebooks/MIPO_NDD_V4_Validation_Colab.ipynb` (upload
`dist/mipo_ndd_v4_colab_v2.zip`) runs this same launcher with the Colab settings names.

Seeds, test genes, config values and code all go into the experiment folder's hash,
as on Colab. If you change them, you start a new experiment. `--gpus`, `--workers`
and `--jobs-per-gpu` only change speed, so you can change them between launches.

## 6. Monitor

- Every 10 minutes the launcher prints each running job's epoch, best validation
  score and bad-epoch count.
- Per-run log: `<experiment>/runs/<model>/seed_<s>/fold_<ff>/train.log`
  (`tail -f` it); per-epoch history: `history.json` in the same folder.
- `nvidia-smi` shows GPU use. The model is small (about 1M trainable parameters),
  so these runs are usually limited by CPU-side graph building, not by the GPU.
  If GPUs sit mostly idle and CPU cores are free, raise `--workers`, or try
  `--jobs-per-gpu 2` and compare the minutes per epoch in the progress lines.

## 7. Stop, resume, report

- **Resume:** rerun exactly the same command. Finished runs are skipped. Interrupted
  runs restart from their last completed epoch. Early-stopped runs stay stopped.
- **Stop:** press Ctrl-C in tmux, or run `scancel <jobid>`. Checkpoints are written
  atomically, so stopping never corrupts them.
- **Reports only:** use the same command plus `--report-only`. It rebuilds the
  tables from whatever has finished, even while training is still running.
- A second launcher on the same experiment or feature cache is refused, not interleaved.

## 8. Results

Everything is in `<root>/v4_validation_<scope>_<size>_<hash>/`:

| File | Content |
|---|---|
| `protocol.json`, `config.json`, `splits.json` | The fixed plan (hashes, config, roles) |
| `homology_report.csv` | Closest training proteins to each test protein |
| `run_status.csv` | Every requested run: complete?, best epoch, stop reason |
| `matched_comparison.csv`, `macro_comparison_by_seed.csv` | Model vs cached LLR on identical test rows |
| `test_diagnostics.csv` | Per-assay test Spearman, bias, sigma, Gaussian and calibrated coverage |
| `runs/<model>/seed_<s>/fold_<ff>/` | Checkpoints, histories, predictions, provenance, `train.log` |
| `launches.jsonl` | One line per launch: arguments, host, GPUs, driver, failures |

Copy results to your PC:

```bash
rsync -av <user>@<dgx-host>:/raid/<user>/mipo_ndd/v4_validation_direct_650M_<hash>/ ./results/
# add --exclude '*.pt' to skip checkpoints
```

Compare models within one experiment folder, trained on one machine. Colab and the
DGX give the same folder name for the same settings, but their GPUs and PyTorch
builds differ. Keep their results under separate roots. If finished runs with
different PyTorch builds end up in one folder, the report prints a warning. Results
from the older Colab release (before this launcher) are in folders with other names.

## 9. Scaling up (decide the plan before looking at test results)

- Replication: `--seeds 42,123,2026` gives 12 runs, which fill 8 GPUs with 4 waiting.
  Adding seeds creates a new experiment folder, but runs already finished in a folder
  that differs only in the seed list (same code, data and settings, same PyTorch build)
  are copied in, not retrained, and their inputs are re-checked. `--no-reuse` turns this off.
- More test genes: `--test-genes PTEN,<gene>,...`. Check `homology_report.csv`
  first. Split groups are single genes, so paralogs can sit in training. For
  example, CHD4's test constructs are 91–94% identical to CHD3/CHD5 in training,
  and DLG4 is about 80% identical to DLG3. Their test scores would be optimistic.
  PTEN has no such homolog in V4. For a strict benchmark, build MMseqs2 clusters
  (`scripts/mmseqs_clusters.py`) and regenerate splits with `mipo splits --clusters`.
  That is a new protocol.
- Curriculum: `--scope curriculum` reuses the same feature cache. PTEN's test rows are the
  same in both scopes, so they compare directly; the validation genes differ. Both scopes:
  `for s in direct curriculum; do bash scripts/dgx/run_v4_validation.sh --root /raid/$USER/mipo_ndd --scope $s --seeds 42,123,2026; done`

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `No CUDA GPU is visible` | You are on a login node. Use `sbatch`/`srun`, or SSH to the GPU node. |
| `CUDA cannot use GPU(s) [...]` | A `--gpus` id does not exist or the GPU is faulty. Compare with `nvidia-smi`; under Slurm, omit `--gpus`. |
| `GPU memory already in use (GB): {...}` | Someone else is using that GPU. Pick free ones from `nvidia-smi` with `--gpus`. Use `--allow-busy-gpus` only if sharing is intended. |
| `refusing to train on CPU` in a `train.log` | That run could not see its GPU. Check `nvidia-smi` (a GPU may have dropped out), then rerun the same command. |
| `Could not create .venv` during setup | `python3-venv` is missing. Use the Miniforge route in section 2. |
| Hugging Face connection error | `HF_HUB_OFFLINE=1` is set but the weights are not in `HF_HOME`. Rerun `setup_env.sh` with the same `HF_HOME` on a node with internet access. |
| `Bus error` / `DataLoader worker ... killed` in a container | Start the container with `--ipc=host` (or `--shm-size=32g`), or use `--workers 0`. |
| `Too many open files` | The run script raises `ulimit -n`. Otherwise run `ulimit -n 65536` first, or lower `--workers`. |
| `CUDA driver version is insufficient` | Reinstall with `TORCH_INDEX=https://download.pytorch.org/whl/cu118`. |
| `Resume provenance mismatch` | The environment (e.g. PyTorch version) changed since that run was interrupted. Restore it, or delete that one run folder to restart it. |
| `Feature cache ... does not match this protocol` | The cache was built with another ESM model or without LLR. Use `--cache` to point at a separate folder. |
| `is in use by another launcher` | Another launch holds the lock. Locks are released when a process exits, so none are ever left behind. |
| `WARNING: cannot lock ...` | The output disk does not support file locks (some NFS mounts). The run continues, but a second launch on the same folder would not be blocked. Prefer `/raid`. |
| A run `FAILED` | Read its `train.log`. Other runs continue. Fix the cause and rerun the same command. |
