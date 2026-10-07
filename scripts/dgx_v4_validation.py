"""MIPO-NDD V4 multi-gene validation on a DGX A100 (or any multi-GPU Linux node).

Command-line replacement for the one-cell Colab launcher (colab_v4_validation.py).
The scientific protocol is unchanged: the same split expansion, configuration,
validation-only checkpoint selection, ablations and exact-row LLR comparison.

Operational differences:
  * independent model/seed/fold runs go to separate GPUs in parallel;
  * data loading uses worker processes (identical batches, higher throughput);
  * the feature cache is checked protein by protein and site by site before any
    training, and is built once for the full V4 corpus so it never changes later;
  * rerunning the same command resumes everything; a lock blocks concurrent launches.

Defaults reproduce the Colab cell. Example (4 GPUs, the default PTEN plan):
    python scripts/dgx_v4_validation.py --root /raid/$USER/mipo_ndd --gpus 0,1,2,3
See DGX_A100.md for setup, Slurm use and troubleshooting.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
# Import this copy's package, never another release installed in the environment.
sys.path.insert(0, str(PROJECT))
# CUDA numbers GPUs fastest-first by default, nvidia-smi and Slurm by PCI bus. Use the
# latter (before CUDA starts) so --gpus means the GPUs nvidia-smi shows as free.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import pandas as pd  # noqa: E402

from mipo.common import digest, save_json  # noqa: E402
from mipo.corpus import read_corpus  # noqa: E402
from mipo.features import cache_status  # noqa: E402
from mipo.splits import expand_holdouts, fold_task_support  # noqa: E402
from scripts.run_experiments import ABLATIONS  # noqa: E402

DATA = PROJECT/"v4/data"
CORPORA = {"direct": DATA/"ndd_direct_v4.csv.gz", "curriculum": DATA/"ndd_training_corpus_v4.csv.gz"}
RESOURCES = PROJECT/"v4/resources"
METADATA = DATA/"assay_metadata.csv"
ESM = {"150M": "facebook/esm2_t30_150M_UR50D", "650M": "facebook/esm2_t33_650M_UR50D"}
# Files that decide what a run computes. The launcher itself is excluded so that
# its arguments, which the protocol records explicitly, never orphan resumable runs.
RUN_SCRIPTS = ["run_experiments.py", "compare_cached_llr.py", "prepare_validation_splits.py"]
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)


def banner(text):
    print("\n"+"="*78+"\n"+text+"\n"+"="*78, flush=True)


def show(frame):
    print(frame.to_string(), flush=True)


def now():
    return datetime.now().isoformat(timespec="seconds")


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.environ.get("MIPO_ROOT", str(PROJECT/"outputs")),
                   help="Persistent output root for features and experiments (default: $MIPO_ROOT or ./outputs)")
    p.add_argument("--scope", choices=["direct", "curriculum"], default="direct")
    p.add_argument("--model-size", choices=list(ESM), default="650M")
    p.add_argument("--no-llr", action="store_true", help="Disable masked-LLR inputs (Colab USE_LLR=False)")
    p.add_argument("--test-genes", default="PTEN", help="Comma-separated. Add genes only as a predefined evaluation plan.")
    p.add_argument("--seeds", default="42", help="Comma-separated model seeds, e.g. 42,123,2026 for replication")
    p.add_argument("--split-seed", type=int, default=42, help="Same validation/calibration split across model seeds")
    p.add_argument("--validation-groups", type=int, default=5)
    p.add_argument("--calibration-groups", type=int, default=3)
    p.add_argument("--steps-per-epoch", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=20, help="Direct scope only; curriculum uses its stage budgets")
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--features", choices=["auto", "build", "reuse"], default="auto",
                   help="auto: build/resume whatever is missing; build: always run (resumable) extraction; reuse: never extract")
    p.add_argument("--feature-scope", choices=["all", "scope"], default="all",
                   help="all: cache every V4 protein/site once, so later scopes never modify it; scope: only this corpus")
    p.add_argument("--cache", help="Feature cache folder (default: ROOT/features/v4_<size>_<masked|no_llr>)")
    p.add_argument("--feature-archive", help="ZIP of an existing cache, e.g. the Colab v4_650M_masked.zip")
    p.add_argument("--local-cache", help="Train from a copy of the verified cache here (for slow storage such as Google Drive)")
    p.add_argument("--ablations", default="esm_mlp,fixed_heads,no_structure",
                   help="Main mipo runs once, even if also listed here")
    p.add_argument("--no-train", action="store_true", help="Skip the main mipo model (Colab DO_TRAIN=False)")
    p.add_argument("--no-ablations", action="store_true")
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--no-reuse", action="store_true",
                   help="Do not copy finished runs from earlier experiments that differ only in seeds or folds")
    p.add_argument("--gpus", help="Comma-separated GPU ids from nvidia-smi (default: every visible GPU)")
    p.add_argument("--jobs-per-gpu", type=int, default=1)
    p.add_argument("--allow-busy-gpus", action="store_true",
                   help="Start even if a chosen GPU already holds over 2 GB of another program's memory")
    p.add_argument("--workers", type=int, help="DataLoader workers per run (default: from available CPUs, at most 8)")
    p.add_argument("--progress-minutes", type=float, default=10., help="How often to print each running job's epoch")
    p.add_argument("--plan-only", action="store_true", help="Fix splits/protocol and check features; no GPU work")
    p.add_argument("--report-only", action="store_true", help="Rebuild reports from existing runs; no GPU work")
    a = p.parse_args(argv)
    a.test_genes = [x.strip() for x in a.test_genes.split(",") if x.strip()]
    a.seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    a.use_llr = not a.no_llr
    a.models = list(dict.fromkeys(([] if a.no_train else ["mipo"]) +
                                  ([] if a.no_ablations else [x.strip() for x in a.ablations.split(",") if x.strip()])))
    problems = []
    if not a.seeds or len(set(a.seeds)) != len(a.seeds):
        problems.append("--seeds must be non-empty and unique")
    if not a.test_genes or len(set(a.test_genes)) != len(a.test_genes):
        problems.append("--test-genes must be non-empty and unique")
    if a.validation_groups < 2 or a.calibration_groups < 1:
        problems.append("need --validation-groups >= 2 and --calibration-groups >= 1")
    if min(a.steps_per_epoch, a.epochs, a.patience, a.jobs_per_gpu) < 1 or a.progress_minutes <= 0 \
            or (a.workers is not None and a.workers < 0):
        problems.append("steps, epochs, patience and jobs per GPU must be positive; workers non-negative")
    unknown = [m for m in a.models if m not in ABLATIONS]
    if unknown:
        problems.append(f"unknown models {unknown}; choose from {list(ABLATIONS)}")
    if a.plan_only and a.report_only:
        problems.append("choose at most one of --plan-only and --report-only")
    if problems:
        p.error("; ".join(problems))
    return a


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def visible_gpus(requested):
    import torch
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    listed = [g.strip() for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()][:count] or \
        [str(i) for i in range(count)]
    if not requested:
        return listed
    chosen = [g.strip() for g in requested.split(",") if g.strip()]
    if os.environ.get("CUDA_VISIBLE_DEVICES") and set(chosen) - set(listed):
        raise SystemExit(f"--gpus {chosen} is outside this allocation's CUDA_VISIBLE_DEVICES={listed}; omit --gpus under Slurm")
    return chosen


def environment():
    import torch
    info = {"host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "cpus": available_cpus(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
            "tf32_cublas_override": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE")}
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=60)
        info["nvidia_smi"] = query.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        info["nvidia_smi"] = None
    return info


def child_env(gpu, threads):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["CUDA_VISIBLE_DEVICES"] = gpu
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        env[name] = str(threads)
    env["PYTHONUNBUFFERED"] = "1"
    return env


PROBE = ("import json, torch\n"
         "ok = torch.cuda.is_available()\n"
         "free, total = torch.cuda.mem_get_info() if ok else (0, 0)\n"
         "print(json.dumps({'ok': ok, 'name': torch.cuda.get_device_name() if ok else None, 'free': free, 'total': total}))")


def probe_gpus(slots):
    """Each GPU as a run will see it: same environment, so bad ids fail here, not hours later."""
    processes = {slot: subprocess.Popen([sys.executable, "-c", PROBE], env=child_env(slot, 1), cwd=PROJECT,
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                 for slot in dict.fromkeys(slots)}
    found = {}
    for slot, process in processes.items():
        try:
            found[slot] = json.loads(process.communicate(timeout=300)[0].strip().splitlines()[-1])
        except (ValueError, IndexError, subprocess.TimeoutExpired):
            process.kill()
            found[slot] = {"ok": False}
    return found


def choose_gpus(gpus, probes, explicit, allow_busy, min_total=10 << 30, max_used=2 << 30):
    """Refuse GPUs a run cannot use or would share; skip small display GPUs unless named explicitly."""
    unusable = [g for g in gpus if not probes[g]["ok"]]
    if unusable:
        raise SystemExit(f"CUDA cannot use GPU(s) {unusable}. Pick ids from nvidia-smi with --gpus, or omit --gpus.")
    small = [g for g in gpus if probes[g]["total"] < min_total]
    if small and explicit:
        raise SystemExit(f"GPU(s) {small} have under {min_total >> 30} GB (for example a DGX Station display GPU); "
                         "choose the A100s with --gpus.")
    if small:
        print(f"Skipping GPU(s) {small} with under {min_total >> 30} GB (for example a display GPU).", flush=True)
    chosen = [g for g in gpus if g not in small]
    # The probe's own CUDA context takes a few hundred MB; more means another program is there.
    busy = {g: round((probes[g]["total"]-probes[g]["free"])/2**30, 1) for g in chosen
            if probes[g]["total"]-probes[g]["free"] > max_used}
    if busy and not allow_busy:
        raise SystemExit(f"GPU memory already in use (GB): {busy}. Check nvidia-smi and choose free GPUs "
                         "with --gpus, or pass --allow-busy-gpus if sharing is intended.")
    if not chosen:
        raise SystemExit("No usable GPU left.")
    return chosen


@contextmanager
def exclusive(path, what):
    """Refuse a second launcher on the same experiment or feature cache."""
    try:
        import fcntl
    except ImportError:  # Windows development machines: no concurrent-launch protection.
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"{what} is in use by another launcher (lock {path}); wait for it or use another --root") from None
        except OSError as error:  # Network/FUSE mounts (NFS, Google Drive) may not support locks.
            print(f"WARNING: cannot lock {path} ({error}); a second launch on this folder would not be blocked.", flush=True)
        yield


def mirror(source, target):
    """Copy a feature cache to faster storage, skipping files already copied unchanged."""
    def copy_changed(src, dst):
        if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)
                and os.path.getmtime(dst) >= os.path.getmtime(src)):
            shutil.copy2(src, dst)
    shutil.copytree(source, target, dirs_exist_ok=True, copy_function=copy_changed,
                    ignore=shutil.ignore_patterns("*.tmp.npy", "*.tmp"))


def release_fingerprint():
    """Code, configs and reference resources that decide what a run computes."""
    paths = [*sorted((PROJECT/"mipo").glob("*.py")), *(PROJECT/"scripts"/name for name in RUN_SCRIPTS),
             *sorted((PROJECT/"configs").glob("v4_*.json")), PROJECT/"pyproject.toml",
             RESOURCES/"sequences.json", *sorted((RESOURCES/"structures").glob("*"))]
    h = hashlib.sha256()
    for path in paths:
        h.update(f"{path.relative_to(PROJECT).as_posix()}\0{digest(path)}\n".encode())
    return h.hexdigest()


def write_once(path, data):
    """Protocol files are fixed by the experiment hash; a differing copy means corruption."""
    if path.exists():
        if path.read_bytes() != data:
            raise SystemExit(f"{path} differs from this protocol; the experiment folder was edited. Restore or remove it.")
    else:
        path.write_bytes(data)


@contextmanager
def _nothing():
    yield


def aligned_identity(aligner, query, target, sites):
    """Identity over aligned columns, and the share of measured query sites (0-based) inside the alignment."""
    alignment = aligner.align(query, target)[0]
    blocks = list(zip(*alignment.aligned))
    same = sum(query[qs+k] == target[ts+k] for (qs, qe), (ts, _) in blocks for k in range(qe-qs))
    span = sum(qe-qs for (qs, qe), _ in blocks)
    covered = sum(any(qs <= s < qe for (qs, qe), _ in blocks) for s in sites)
    return same/max(span, 1), span/len(query), covered/max(len(sites), 1)


def homology_report(corpus, split, folds, min_identity=.3, min_coverage=.5):
    """Closest training proteins to each test protein. Split groups are single genes,
    so paralogs of a test gene can sit in training; PTEN has none in V4. Coverage
    counts measured test sites, because training constructs are often single domains."""
    from Bio import Align
    from Bio.Align import substitution_matrices
    aligner = Align.PairwiseAligner(mode="local", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                                    open_gap_score=-11, extend_gap_score=-1)
    sequences = json.loads((RESOURCES/"sequences.json").read_text())
    genes = corpus.groupby("protein_reference_id").gene.first()
    measured = corpus.groupby("protein_reference_id").ref_pos.unique()
    rows = []
    for i in folds:
        fold = split["folds"][i]
        train = genes[genes.isin(fold["train"])]
        for reference, gene in genes[genes.isin(fold["test"])].items():
            query = sequences[reference]["sequence"]
            scores = sorted(((aligner.score(query, sequences[r]["sequence"]), r) for r in train.index), reverse=True)[:5]
            for score, other in scores:
                identity, coverage, site_coverage = aligned_identity(
                    aligner, query, sequences[other]["sequence"], measured[reference]-1)
                rows.append({"fold": i, "test_gene": gene, "test_reference": reference, "train_gene": train[other],
                             "train_reference": other, "local_score": score, "identity": round(identity, 3),
                             "test_coverage": round(coverage, 3), "measured_site_coverage": round(site_coverage, 3),
                             "possible_homolog": identity >= min_identity and site_coverage >= min_coverage})
    return pd.DataFrame(rows)


def unpack_cache(archive, cache):
    with zipfile.ZipFile(archive) as z:
        manifests = [n for n in z.namelist() if n.split("/")[-1] == "manifest.json" and n.count("/") <= 1]
        if len(manifests) != 1:
            raise SystemExit(f"{archive} must contain one manifest.json at its root or inside one folder")
        prefix = manifests[0][:-len("manifest.json")]
        base = cache.resolve()
        base.mkdir(parents=True, exist_ok=True)
        for item in z.infolist():
            if item.is_dir() or not item.filename.startswith(prefix):
                continue
            target = (base/item.filename[len(prefix):]).resolve()
            if base not in target.parents:
                raise SystemExit("Unsafe path in feature archive")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(item) as source, open(target, "wb") as sink:
                shutil.copyfileobj(source, sink)


def reusable_runs(root, experiment, protocol, jobs, torch_version):
    """Finished runs in earlier experiments whose protocol differs only in the seed list or folds.

    Each run depends on its own seed and fold, not on which others were requested, so
    such runs are the same computation. run_experiments.py re-checks every input hash
    before accepting a copied run; runs from another PyTorch build are left out.
    """
    def core(p):
        p = copy.deepcopy(p)
        p.pop("seeds", None)
        p.pop("folds", None)
        p["config"].pop("seed", None)
        return p
    found = {}
    prefix = experiment.name.rsplit("_", 1)[0]
    for other in sorted(root.glob(prefix+"_*")):
        if other == experiment or not (other/"protocol.json").exists():
            continue
        if core(json.loads((other/"protocol.json").read_text())) != core(protocol):
            continue
        for job in jobs:
            source = run_folder(other/"runs", *job)
            if job in found or not all((source/name).exists() for name in
                                       ["test_summary.json", "training_status.json", "provenance.json"]):
                continue
            if json.loads((source/"provenance.json").read_text()).get("torch") == torch_version:
                found[job] = source
    return found


def adopt_runs(found, runs, copy_files=True):
    """Copy reusable runs in; a run already started or finished here takes precedence."""
    for job, source in found.items():
        target = run_folder(runs, *job)
        if target.exists():
            continue
        if copy_files:  # Copy under a temporary name, so an interrupted copy is never mistaken for a run.
            partial = target.with_name(target.name+".copying")
            shutil.rmtree(partial, ignore_errors=True)
            shutil.copytree(source, partial)
            partial.rename(target)
        print(f"{'Reused' if copy_files else 'Will reuse'} finished run {label(job)} from {source.parents[3].name}",
              flush=True)


def run_folder(runs, model, seed, fold):
    return Path(runs)/model/f"seed_{seed}"/f"fold_{fold:02d}"


def label(job):
    model, seed, fold = job
    return f"{model} seed={seed} fold={fold}"


def progress(folder):
    try:
        history = json.loads((folder/"history.json").read_text())
    except (OSError, ValueError):
        return "starting (loading data / first epoch)"
    last = history[-1]
    best = max(h["validation_spearman"] for h in history)
    return f"epoch {last['epoch']+1} ({last['stage']}), best validation {best:.4f}, bad epochs {last['bad_epochs']}"


def log_tail(path, count=6):
    """The last output lines of a run, where its error message is."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [line for line in lines if line.strip() and not line.startswith("=====")][-count:]


def dispatch(jobs, slots, command, log_path, env, poll=15., report_every=600., healthy=lambda slot: True):
    """Run independent jobs, one per slot at a time. Failures do not stop other jobs.

    Returns {job: exit code}; None marks a job never started. After a failure the
    slot is checked, so a GPU that stopped working cannot fail the whole queue.
    On interruption every child is stopped; runs resume from their last completed
    epoch next time.
    """
    pending, running, results, free = list(jobs), {}, {}, list(slots)
    last_report = time.time()
    try:
        while pending or running:
            if pending and not free and not running:
                print(f"No working GPU left; not started: {[label(job) for job in pending]}", flush=True)
                results.update({job: None for job in pending})
                break
            while pending and free:
                job, slot = pending.pop(0), free.pop(0)
                path = log_path(job)
                path.parent.mkdir(parents=True, exist_ok=True)
                log = open(path, "a", encoding="utf-8")
                args = [str(x) for x in command(job)]
                log.write(f"\n===== {now()} start on GPU '{slot}': {' '.join(args)}\n")
                log.flush()
                process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env(slot), cwd=PROJECT)
                running[process] = (job, slot, log, time.time())
                print(f"[start] {label(job)} on GPU '{slot}'; log {path}", flush=True)
            time.sleep(poll)
            for process in [p for p in running if p.poll() is not None]:
                job, slot, log, began = running.pop(process)
                log.write(f"===== {now()} exit {process.returncode}\n")
                log.close()
                if process.returncode == 0 or healthy(slot):
                    free.append(slot)
                else:
                    print(f"GPU '{slot}' no longer works; its remaining jobs go to other GPUs.", flush=True)
                results[job] = process.returncode
                state = "done" if process.returncode == 0 else f"FAILED (exit {process.returncode}); see {log_path(job)}"
                print(f"[{now()}] {label(job)}: {state} after {(time.time()-began)/60:.1f} min", flush=True)
                if process.returncode != 0:
                    print("\n".join("    "+line for line in log_tail(log_path(job))), flush=True)
            if running and time.time()-last_report >= report_every:
                last_report = time.time()
                for job, *_ in running.values():
                    print(f"[{now()}] {label(job)}: {progress(log_path(job).parent)}", flush=True)
    except BaseException:
        for process in running:
            process.terminate()
        for process in running:
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    return results


def run_table(runs, models, seeds, folds):
    records = []
    for model in models:
        for seed in seeds:
            for fold in folds:
                folder = run_folder(runs, model, seed, fold)
                record = {"model": model, "seed": seed, "fold": fold, "folder": str(folder),
                          "complete": (folder/"training_status.json").exists() and (folder/"test_metrics.csv").exists()}
                if (folder/"training_status.json").exists():
                    record.update(json.loads((folder/"training_status.json").read_text()))
                if (folder/"provenance.json").exists():
                    record["torch"] = json.loads((folder/"provenance.json").read_text()).get("torch")
                elif (folder/"history.json").exists():
                    record["epochs_run"] = len(json.loads((folder/"history.json").read_text()))
                records.append(record)
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["model", "seed", "fold", "folder", "complete"])


def collect(status, filename):
    frames = []
    for run in status[status.complete].itertuples():
        path = Path(run.folder)/filename
        if path.exists():
            frame = pd.read_csv(path)
            frame.insert(0, "model", run.model)
            frame.insert(1, "seed", run.seed)
            frame.insert(2, "fold", run.fold)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def common_evaluation(table, models):
    """Keep seed/fold pairs that every model finished, so macro scores average the same genes."""
    finished = {m: set(zip(g.seed, g.fold)) for m, g in table.groupby("model")}
    common = set.intersection(*(finished.get(m, set()) for m in models)) if models else set()
    dropped = {m: sorted(s-common) for m, s in finished.items() if s-common}
    keep = table[[pair in common for pair in zip(table.seed, table.fold)]]
    return keep, common, dropped


def report(a, runs, experiment, cache, folds, env_for_tools):
    models = a.models or sorted(p.name for p in runs.glob("*") if p.is_dir())
    banner("5. TRAINING HISTORY AND VALIDATION DIAGNOSTICS")
    status = run_table(runs, models, a.seeds, folds)
    status.to_csv(experiment/"run_status.csv", index=False)
    builds = sorted(status.loc[status.complete, "torch"].dropna().unique()) if "torch" in status else []
    if len(builds) > 1:  # Colab and DGX give the same folder name for the same settings.
        print(f"WARNING: finished runs used different PyTorch builds {builds}, e.g. some on Colab and some on "
              "the DGX. Model differences may partly reflect the machine; compare runs from one environment.")
    for run in status.itertuples():
        folder = Path(run.folder)
        print(f"\n{folder.relative_to(runs)}")
        if not run.complete:
            print("  INCOMPLETE: rerun the same command to resume it"
                  + (f" ({int(run.epochs_run)} epochs saved)" if pd.notna(getattr(run, "epochs_run", None)) else ""))
            continue
        history = pd.DataFrame(json.loads((folder/"history.json").read_text()))
        losses = pd.json_normalize(history.loss_components).add_prefix("train_")
        show(pd.concat([history[["epoch", "loss", "validation_spearman", "lr", "bad_epochs"]], losses], axis=1).round(5))
        print("Saved checkpoint / stopping:", json.loads((folder/"training_status.json").read_text()))
        show(pd.DataFrame(history.validation_per_gene.tolist(), index=history.epoch).round(4))
        print("Best checkpoint validation by gene and assay:")
        validation = pd.read_csv(folder/"validation_metrics.csv")
        show(validation[["gene", "task", "n", "spearman", "bias", "mean_sigma", "coverage_90_gaussian"]].round(4))
    print("The history describes this validation panel; it does not establish the cause of a plateau.")

    if not a.no_baseline:
        banner("6. EXACT-ROW CACHED LLR BASELINE")
        if not a.use_llr:
            print("Skipped: this protocol has no masked-LLR cache to compare against.")
        elif not status.complete.any():
            print("Skipped: no completed runs yet.")
        else:
            subprocess.run([sys.executable, str(PROJECT/"scripts/compare_cached_llr.py"), "--corpus", str(CORPORA[a.scope]),
                            "--resources", str(RESOURCES), "--cache", str(cache), "--runs", str(runs)],
                           check=True, env=env_for_tools, cwd=PROJECT)

    banner("7. TEST RANKING AND INTERVAL DIAGNOSTICS")
    comparisons = collect(status, "matched_llr_metrics.csv")
    if not comparisons.empty:
        comparisons.to_csv(experiment/"matched_comparison.csv", index=False)
        show(comparisons.round(4))
        keep, common, dropped = common_evaluation(comparisons, [m for m in models if m in set(comparisons.model)])
        missing = [m for m in models if m not in set(comparisons.model)]
        if missing or dropped:
            print(f"Unequal evaluation sets. Without results: {missing}; seed/fold pairs outside the common set: {dropped}.")
            print(f"The macro comparison below uses only seed/fold pairs every reported model finished: {sorted(common)}")
        if keep.empty:
            print("No seed/fold pair is finished by every model; headline macro comparison suppressed.")
        elif keep[["model_spearman", "llr_spearman"]].notna().all().all():
            gene_scores = keep.groupby(["model", "seed", "gene"])[["model_spearman", "llr_spearman", "delta"]].mean()
            per_seed = gene_scores.groupby(["model", "seed"]).mean()
            print("Macro across genes, separately for each seed (all assays retained):")
            show(per_seed.round(4))
            per_seed.to_csv(experiment/"macro_comparison_by_seed.csv")
            if len(a.seeds) > 1:
                print("Mean and standard deviation across seeds; this is training variability, not uncertainty across genes:")
                show(per_seed.groupby("model").agg(["mean", "std"]).round(4))
        else:
            print("Some correlations are undefined: inspect the assay table; headline macro comparison suppressed.")
    intervals = collect(status, "test_metrics.csv")
    if not intervals.empty:
        intervals.to_csv(experiment/"test_diagnostics.csv", index=False)
        show(intervals[["model", "seed", "gene", "task", "n", "spearman", "bias", "mean_sigma",
                        "coverage_90_gaussian", "coverage_calibrated", "calibrated_interval_rows"]].round(4))
        print("Gaussian coverage uses raw predicted sigma. Calibrated coverage uses separate calibration groups.")
        print("Missing calibrated coverage means no supported calibration interval. Nominal coverage is 0.90; transfer to unseen genes is empirical.")
        print("Bias, sigma, RMSE and NLL are in training-task-transformed units. Raw-score predictions and intervals are also saved.")
    print(f"Evaluated {len(folds)} outer folds and {len(a.seeds)} seeds. Test results must not select checkpoints or tune this protocol.")
    return status


def main(argv=None):
    a = arguments(argv)
    root = Path(a.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    corpus_path = CORPORA[a.scope]
    gpu_work = not (a.plan_only or a.report_only)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # Slurm time limit / scancel: stop children cleanly.

    banner("1. ENVIRONMENT")
    env = environment()
    for key, value in env.items():
        print(f"{key}: {value}")
    gpus = visible_gpus(a.gpus) if env["cuda_available"] or a.gpus else []
    if gpu_work and not (env["cuda_available"] and gpus):
        raise SystemExit("No CUDA GPU is visible. On a cluster, run on a GPU node or inside your Slurm GPU allocation; "
                         "on Colab, choose Runtime > Change runtime type > GPU.")
    if gpu_work:
        probes = probe_gpus(gpus)
        for gpu, info in probes.items():
            print(f"GPU '{gpu}': {info.get('name')}, {info.get('free', 0)/2**30:.1f} of "
                  f"{info.get('total', 0)/2**30:.1f} GB free" if info["ok"] else f"GPU '{gpu}': NOT USABLE")
        gpus = choose_gpus(gpus, probes, bool(a.gpus), a.allow_busy_gpus)
        env["gpu_probe"] = probes
    print("GPUs used:", gpus or "none")

    banner("2. FIX SPLITS AND CHECKPOINT SELECTION BEFORE TRAINING")
    corpus = read_corpus(corpus_path)
    staging = Path(tempfile.mkdtemp(prefix=".staging_", dir=root))
    try:
        staged = staging/"splits.json"
        split = expand_holdouts(corpus_path, PROJECT/f"v4/splits/{a.scope}.json", staged,
                                a.validation_groups, a.calibration_groups, a.split_seed)
        folds = sorted({i for i, fold in enumerate(split["folds"]) if set(a.test_genes) & set(fold["test"])})
        absent = set(a.test_genes) - {g for i in folds for g in split["folds"][i]["test"]}
        if absent:
            raise SystemExit(f"Test genes absent from outer folds: {sorted(absent)}")
        if any(split["folds"][i]["fold"] != i for i in folds):
            raise SystemExit("Fold numbering differs from fold order; runs and reports would disagree")
        for i in folds:
            fold = split["folds"][i]
            print(f"Fold {i}: test={fold['test']}, validation={fold['validation']}, "
                  f"calibration={fold['calibration']}, train genes={len(fold['train'])}")
            counts = fold_task_support(corpus, fold)
            print("Number of genes per task in each role:")
            show(pd.DataFrame(counts).T)
            for task, roles in counts.items():
                if roles["test"] and (not roles["validation"] or not roles["calibration"]):
                    print(f"Task coverage gap: {task}. Validation/calibration transfer for this task is not established.")
        config = json.loads((PROJECT/f"configs/v4_{a.scope}.json").read_text())
        config.update(seed=a.seeds[0], use_llr=a.use_llr, require_llr=a.use_llr,
                      steps_per_epoch=a.steps_per_epoch, patience=a.patience,
                      selection_metric="macro_gene_spearman", min_validation_genes=a.validation_groups)
        if "stages" in config:
            print("Curriculum stage budgets:", [(s["name"], s["epochs"]) for s in config["stages"]])
            max_epochs = sum(s["epochs"] for s in config["stages"])
        else:
            config["epochs"] = a.epochs
            max_epochs = a.epochs
        protocol = {"code_and_resources_sha256": release_fingerprint(), "corpus_sha256": digest(corpus_path),
                    "split_sha256": digest(staged), "metadata_sha256": digest(METADATA), "config": config,
                    "model_size": a.model_size, "seeds": a.seeds, "folds": folds}
        protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
        experiment = root/f"v4_validation_{a.scope}_{a.model_size}_{protocol_hash}"
        if a.report_only and not experiment.exists():
            raise SystemExit(f"No experiment folder for these settings: {experiment}. Use the same arguments as the training launch.")
        experiment.mkdir(parents=True, exist_ok=True)
        write_once(experiment/"splits.json", staged.read_bytes())
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    write_once(experiment/"config.json", json.dumps(config, indent=2).encode())
    write_once(experiment/"protocol.json", json.dumps(protocol, indent=2).encode())
    runs = experiment/"runs"
    print("Outputs:", experiment)
    print(f"{len(a.models)*len(a.seeds)*len(folds)} requested model/seed/fold runs: {a.models}")
    print(f"{a.steps_per_epoch} batches x {config['batch_size']} = {a.steps_per_epoch*config['batch_size']:,} "
          f"sample draws/epoch; max {max_epochs} epochs; patience {a.patience}")
    print("Selection: mean assay Spearman within each validation gene, then equal mean across genes.")
    print("Expanded holdouts change training data; compare models within this new protocol.")
    homology = experiment/"homology_report.csv"
    if not homology.exists():
        homology_report(corpus, split, folds).to_csv(homology, index=False)
    flagged = pd.read_csv(homology).query("possible_homolog")
    if flagged.empty:
        print("Homology check: no training protein reaches 30% identity over 50% of any test protein's measured sites.")
    else:
        print("WARNING: training contains possible homologs of test proteins; the test score may be optimistic:")
        show(flagged)

    with exclusive(experiment/".lock", "This experiment") if not a.report_only else _nothing():
        banner("3. REUSE OR EXTRACT FROZEN FEATURES")
        model_name = ESM[a.model_size]
        tag = f"v4_{a.model_size}_{'masked' if a.use_llr else 'no_llr'}"
        cache = Path(a.cache).resolve() if a.cache else root/"features"/tag
        feature_corpus = read_corpus(CORPORA["curriculum"]) if a.feature_scope == "all" and a.scope != "curriculum" else corpus
        with exclusive(cache.parent/f".{cache.name}.lock", "This feature cache"):
            if a.feature_archive and not (cache/"manifest.json").exists():
                unpack_cache(Path(a.feature_archive), cache)
                print("Feature archive extracted:", cache)
            wanted = cache_status(feature_corpus, RESOURCES, cache, a.use_llr, model_name)
            if wanted["incompatible"]:
                raise SystemExit(f"Feature cache {cache} does not match this protocol: {wanted['incompatible']}. Use another --cache.")
            print(f"Cache {cache}: {wanted['references']-len(wanted['missing_embeddings'])}/{wanted['references']} "
                  f"proteins embedded, {wanted['missing_llr_sites']} LLR sites missing ({a.feature_scope} scope)")
            build = a.features == "build" or (a.features == "auto" and not wanted["complete"])
            if build and gpu_work:
                print("Building/resuming ESM embeddings and masked LLR on GPU", gpus[0], flush=True)
                command = [sys.executable, "-m", "mipo", "features", "--resources", str(RESOURCES), "--cache", str(cache),
                           "--model", model_name, "--corpus",
                           str(CORPORA["curriculum"] if a.feature_scope == "all" else corpus_path)]
                subprocess.run(command + ([] if a.use_llr else ["--no-llr"]), check=True, cwd=PROJECT,
                               env=child_env(gpus[0], max(1, available_cpus()//2)))
            ready = cache_status(corpus, RESOURCES, cache, a.use_llr, model_name)
            if ready["complete"]:
                print("Feature cache complete for this corpus.")
            elif gpu_work or a.report_only:
                raise SystemExit(f"Feature cache incomplete for this corpus ({len(ready['missing_embeddings'])} proteins, "
                                 f"{ready['missing_llr_sites']} LLR sites missing). Use --features auto or build.")
            else:
                print("Features will be built (or resumed) when you launch without --plan-only.")
        if a.local_cache and ready["complete"] and Path(a.local_cache).resolve() != cache:
            # Same bytes, so run provenance (content hashes, not paths) is unaffected.
            local = Path(a.local_cache).resolve()
            mirror(cache, local)
            if not cache_status(corpus, RESOURCES, local, a.use_llr, model_name)["complete"]:
                raise SystemExit(f"Local cache copy {local} is incomplete; delete it and rerun")
            cache = local
            print("Training reads the local copy:", cache)
        # The runs load their own copy; holding ~1 GB here for hours starves small machines such as Colab.
        del corpus, feature_corpus
        gc.collect()

        banner("4. TRAIN AND RUN MATCHED ABLATIONS")
        jobs = [(model, seed, fold) for seed in a.seeds for fold in folds for model in a.models]
        slots = [gpu for gpu in gpus for _ in range(a.jobs_per_gpu)]
        concurrency = max(1, min(len(jobs), len(slots)))
        per_job = max(1, available_cpus()//concurrency)
        workers = a.workers if a.workers is not None else max(0, min(8, per_job-1))
        threads = max(1, min(4, per_job-workers))
        print(f"{len(jobs)} runs on {len(slots)} GPU slots ({concurrency} at a time); "
              f"{workers} data workers and {threads} CPU threads per run")
        if not (a.no_reuse or a.report_only):
            adopt_runs(reusable_runs(root, experiment, protocol, jobs, env["torch"]), runs, copy_files=gpu_work)
        results = {}
        if jobs and gpu_work:
            started = time.time()

            def command(job):
                model, seed, fold = job
                return [sys.executable, PROJECT/"scripts/run_experiments.py", "--corpus", corpus_path, "--resources", RESOURCES,
                        "--cache", cache, "--splits", experiment/"splits.json", "--config", experiment/"config.json",
                        "--metadata", METADATA, "--out", runs, "--folds", fold, "--seeds", seed,
                        "--ablations", model, "--workers", workers, "--require-cuda"]
            results = dispatch(jobs, slots, command, lambda job: run_folder(runs, *job)/"train.log",
                               lambda slot: child_env(slot, threads), report_every=a.progress_minutes*60,
                               healthy=lambda slot: probe_gpus([slot])[slot]["ok"])
            print(f"Run time: {(time.time()-started)/60:.1f} min")
        elif not gpu_work:
            print("No training in --plan-only / --report-only mode.")
        if a.plan_only:
            banner(f"PLAN READY: {experiment}\nRerun without --plan-only to build features and train.")
            return
        failed = [label(job) for job, code in results.items() if code != 0]
        if gpu_work:
            with open(experiment/"launches.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"finished": now(), "argv": sys.argv[1:], "environment": env, "gpus": gpus,
                                         "workers": workers, "failed": failed}) + "\n")
        status = report(a, runs, experiment, cache, folds, child_env("", 4))
    banner(f"DONE: {experiment}")
    if failed:
        print("Failed runs (fix the cause, then rerun the same command to resume):", failed)
    if gpu_work and not status.complete.all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
