from __future__ import annotations

import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .common import TASKS, digest, save_json, seed_all, seq_key, to_device, torch_save
from .corpus import read_corpus
from .data import BalancedAssayBatches, FeatureStore, Metadata, TargetTransform, VariantDataset, collate
from .losses import loss_function
from .metrics import add_intervals, calibration_quantiles, metrics
from .model import MIPO
from .splits import fold_task_support, verify_fold


def code_fingerprints():
    return {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def selection_score(report, expected_genes):
    """Fixed assay-within-gene, then equal-gene average; never drop a bad gene."""
    if report["genes_with_defined_spearman"] != expected_genes:
        raise ValueError("Validation Spearman undefined for one or more genes; inspect validation metrics")
    value = report["macro_gene_spearman"]
    if value is None or not math.isfinite(value):
        raise ValueError("Validation Spearman must be finite")
    return value


def loader(rows, store, transform, metadata, config, training=False, seed=0, workers=0):
    ds = VariantDataset(rows, store, transform, metadata, config, load_aux=training)
    # Items are pure functions of their row and the sampler runs in this process,
    # so worker processes change throughput, never batch contents or order.
    parallel = {"num_workers": workers, "persistent_workers": workers > 0}
    if training:
        sampler = BalancedAssayBatches(ds.rows, config["batch_size"], config["steps_per_epoch"], seed)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate, **parallel)
    return DataLoader(ds, batch_size=config["batch_size"], shuffle=False, collate_fn=collate, **parallel)


@torch.inference_mode()
def predict(model, batches, device):
    model.eval()
    output = []
    for b in batches:
        b = to_device(b, device)
        pred = model(b)
        output.append(pd.DataFrame({"row_id": b["row_id"].cpu().numpy(), "y": b["y"].cpu().numpy(),
                                    "mu": pred["mu"].float().cpu().numpy(), "sigma": pred["sigma"].float().cpu().numpy()}))
    p = pd.concat(output, ignore_index=True)
    rows = batches.dataset.rows
    p = p.merge(rows[["row_id", "gene", "assay_key", "variant_key", "task", "score_value"]], on="row_id", validate="one_to_one")
    p["task_seen_in_training"] = p.task.map(dict(zip(TASKS, model.seen_tasks.cpu().tolist())))
    return p


def freeze_for_stage(model, freeze):
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if freeze:
        for module in [model.project, model.sequence, model.layers, model.aa, model.impulse, model.impulse_gate]:
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def select_stage_rows(rows, stage):
    selected = rows if '*' in stage['tiers'] else rows[rows.supervision_tier.isin(stage['tiers'])]
    status = stage.get('ndd_filter', 'any')
    if status not in ['any', 'verified', 'unverified']:
        raise ValueError('ndd_filter must be any, verified, or unverified')
    if status != 'any':
        if 'is_direct_ndd' not in selected:
            raise ValueError('This curriculum requires explicit NDD evidence flags')
        direct = pd.to_numeric(selected.is_direct_ndd, errors='raise') == 1
        selected = selected[direct if status == 'verified' else ~direct]
    return selected


def train(corpus, resources, cache, split_file, config_file, out, fold_index=0, metadata_path=None, resume=False, allow_synthetic=False, workers=0):
    config = json.loads(Path(config_file).read_text())
    d = read_corpus(corpus)
    split = json.loads(Path(split_file).read_text())
    if digest(corpus) != split["corpus_sha256"]:
        raise ValueError("Corpus changed after split creation; regenerate splits")
    fold = split["folds"][fold_index]
    verify_fold(d, fold, split.get("gene_to_group"))
    if config.get("selection_metric", "macro_gene_spearman") != "macro_gene_spearman":
        raise ValueError("Only the predefined macro_gene_spearman selection rule is supported")
    if len(fold["validation"]) < config.get("min_validation_genes", 1):
        raise ValueError("Too few validation genes; prepare multi-group splits first")
    if config.get("patience", 5) < 1:
        raise ValueError("Patience must be positive")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.pt").exists() and not resume:
        raise ValueError("Run exists: use --resume or a new output directory")
    seed = config.get("seed", 42)
    seed_all(seed)
    if config.get("cpu_threads"):
        torch.set_num_threads(config["cpu_threads"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    store = FeatureStore(resources, cache, allow_synthetic)
    # Alternate gene names must not place an identical reference in two roles.
    role_by_sequence = {}
    for role in ["train", "validation", "calibration", "test"]:
        for gene in d.loc[d.gene.isin(fold[role]), "protein_reference_id"].unique():
            sequence = store.records[gene]["sequence"]
            previous = role_by_sequence.setdefault(sequence, role)
            if previous != role:
                raise ValueError("Identical protein sequences cross split roles; use a common cluster")
    train_rows = d[d.gene.isin(fold["train"])].copy()
    transform = TargetTransform().fit(train_rows)
    metadata = Metadata(metadata_path).fit(train_rows)
    seen = [False]*len(TASKS)
    model = MIPO(store.manifest["dim"], len(metadata.vocab), config, seen).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", .01))
    stages = config.get("stages", [{"name": "direct", "epochs": config.get("epochs", 12), "tiers": ["*"]}])
    if not stages or any(s["epochs"] < 1 for s in stages):
        raise ValueError("Curriculum stages require positive epochs")
    if stages[0].get("freeze_backbone", False):
        raise ValueError("Cannot freeze a randomly initialized backbone in the first stage")
    # Fail up front instead of reaching a nonexistent teacher stage hours later.
    for stage in stages:
        if select_stage_rows(train_rows, stage).empty:
            raise ValueError(f"No training rows for stage {stage['name']}; manifest plans are not data")
    epochs = sum(s["epochs"] for s in stages)
    accumulation = config.get("grad_accum", 1)
    if accumulation < 1 or config["steps_per_epoch"] < 1:
        raise ValueError("Positive accumulation and steps_per_epoch required")
    updates = epochs*math.ceil(config["steps_per_epoch"]/accumulation)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, updates, eta_min=config["lr"]*.05)
    amp = device == "cuda" and config.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    provenance = {"corpus_sha256": digest(corpus), "split_sha256": digest(split_file), "fold": fold,
                  "config": config, "feature_manifest": store.manifest, "code_sha256": code_fingerprints(),
                  "sequence_sha256": digest(Path(resources)/"sequences.json"),
                  "metadata_sha256": digest(metadata_path) if metadata_path else None,
                  "synthetic_smoke_only": bool(allow_synthetic), "torch": str(torch.__version__), "python": platform.python_version()}
    provenance["feature_files"] = feature_fingerprints(store, d.protein_reference_id.unique())
    provenance["auxiliary_files"] = {}
    if config.get("auxiliary_dir"):
        aux_dir = Path(config["auxiliary_dir"])
        if not (aux_dir/"provenance.json").exists():
            raise ValueError("Auxiliary directory needs provenance from prepare_field_teachers.py")
        for key in train_rows.variant_key.unique():
            path = aux_dir/f"{seq_key(key)}.npz"
            if path.exists():
                provenance["auxiliary_files"][key] = digest(path)
        if not provenance["auxiliary_files"]:
            raise ValueError("No physical teachers match training variants in this fold")
    start, best, bad_epochs, history = 0, -float("inf"), 0, []
    if resume:
        checkpoint = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["provenance"] != provenance:
            raise ValueError("Resume provenance mismatch; restore identical inputs/config/environment")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, best, bad_epochs, history = checkpoint["epoch"]+1, checkpoint["best"], checkpoint["bad_epochs"], checkpoint["history"]
    save_json(out / "provenance.json", provenance)
    save_json(out / "target_transform.json", transform.state)
    composition = []
    for role in ["train", "validation", "calibration", "test"]:
        counts = d[d.gene.isin(fold[role])].groupby(["gene", "assay_key", "task"]).size().reset_index(name="rows")
        counts.insert(0, "role", role)
        composition.append(counts)
    pd.concat(composition, ignore_index=True).to_csv(out / "split_composition.csv", index=False)
    support = fold_task_support(d, fold)
    save_json(out / "task_support.json", support)
    for task, counts in support.items():
        if counts["test"] and (not counts["validation"] or not counts["calibration"]):
            print(f"Task coverage gap ({task}), genes by role: {counts}. Validation/calibration transfer for this task is not established.", flush=True)
    (out / "pip_freeze.txt").write_text(subprocess.check_output([__import__('sys').executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    valid_loader = loader(d[d.gene.isin(fold["validation"])], store, transform, metadata, config, workers=workers)
    print(f"device={device}; trainable_parameters={sum(p.numel() for p in model.parameters()):,}; test={fold['test']}", flush=True)
    previous_stage = None
    already_stopped = bool(history and history[-1]["stage"] == stages[-1]["name"]
                           and bad_epochs >= config.get("patience", 5))
    if already_stopped:
        print("Run already early-stopped; evaluating the saved best checkpoint.", flush=True)
        start = epochs
    for epoch in range(start, epochs):
        seed_all(seed+epoch)  # Epoch-boundary resume reproduces sampler and dropout RNG.
        boundary = 0
        for stage in stages:
            boundary += stage["epochs"]
            if epoch < boundary:
                break
        if stage["name"] != previous_stage:
            freeze_for_stage(model, stage.get("freeze_backbone", False))
            selected = select_stage_rows(train_rows, stage)
            model.seen_tasks |= torch.tensor([task in set(selected.task) for task in TASKS], device=device)
            train_loader = loader(selected, store, transform, metadata, config, True, seed, workers)
            previous_stage = stage["name"]
        train_loader.batch_sampler.epoch = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        components = {}
        for step, b in enumerate(train_loader):
            b = to_device(b, device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                output = model(b)
                loss, parts = loss_function(output, b, config.get("ranking_weight", .2), config.get("field_weight", 1e-5), config.get("auxiliary_weight", .1))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in epoch {epoch}, step {step}")
            group_start = (step//accumulation)*accumulation
            group_size = min(accumulation, len(train_loader)-group_start)
            scaler.scale(loss/group_size).backward()
            if (step+1) % accumulation == 0 or step+1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("clip_grad", 1.))
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            for name, value in parts.items():
                components.setdefault(name, []).append(float(value))
        validation = predict(model, valid_loader, device)
        per_assay, report = metrics(validation)
        diagnostics = out / "validation_epochs"
        diagnostics.mkdir(exist_ok=True)
        per_assay.to_csv(diagnostics / f"epoch_{epoch:03d}.csv", index=False)
        metric = selection_score(report, len(fold["validation"]))
        improved = metric > best
        best, bad_epochs = (metric, 0) if improved else (best, bad_epochs+1)
        row = {"epoch": epoch, "stage": stage["name"], "loss": float(np.mean(losses)), "validation_spearman": metric,
               "lr": optimizer.param_groups[0]["lr"], "improved": improved, "bad_epochs": bad_epochs,
               "loss_components": {k: float(np.mean(v)) for k, v in components.items()},
               "validation_per_gene": report["per_gene"], "validation_per_task": report["per_task"],
               "sample_draws": (epoch+1)*config["steps_per_epoch"]*config["batch_size"]}
        history.append(row)
        save_json(out / "history.json", history)
        print(json.dumps({k: v for k, v in row.items() if k not in ["validation_per_gene", "validation_per_task"]}), flush=True)
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "bad_epochs": bad_epochs, "history": history,
                      "provenance": provenance, "transform": transform.state, "metadata_vocab": metadata.vocab}
        torch_save(out / "last.pt", checkpoint)
        if improved:
            torch_save(out / "best.pt", checkpoint)
            validation.to_csv(out / "validation_predictions.csv", index=False)
            per_assay.to_csv(out / "validation_metrics.csv", index=False)
            save_json(out / "validation_summary.json", report)
        # Complete earlier stages; early stopping applies only in the final stage.
        if stage is stages[-1] and bad_epochs >= config.get("patience", 5):
            break
    best_state = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    save_json(out / "training_status.json", {"epochs_run": len(history), "max_epochs": epochs,
              "best_epoch": best_state["epoch"], "best_validation_spearman": best_state["best"],
              "stop_reason": "early_stopping" if bad_epochs >= config.get("patience", 5) else "epoch_cap",
              "selection_metric": "macro_gene_spearman", "patience": config.get("patience", 5)})
    evaluate(corpus, resources, cache, out / "best.pt", out, metadata_path, allow_synthetic, workers)
    return out


def restore(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state["provenance"]["config"]
    model = MIPO(state["provenance"]["feature_manifest"]["dim"], len(state["metadata_vocab"]), config).to(device)
    model.load_state_dict(state["model"])
    return model, state


def feature_fingerprints(store, genes):
    fingerprints = {}
    for gene in sorted(genes):
        key = store.records[gene]["key"]
        paths = {"esm": store.cache/key/"esm.npy", "llr": store.cache/key/"llr.npy",
                 "structure": store.resources/"structures"/f"{key}.npz"}
        fingerprints[gene] = {name: digest(path) if path.exists() else None for name, path in paths.items()}
    return fingerprints


def evaluate(corpus, resources, cache, checkpoint, out, metadata_path=None, allow_synthetic=False, workers=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, state = restore(checkpoint, device)
    provenance = state["provenance"]
    if digest(corpus) != provenance["corpus_sha256"]:
        raise ValueError("Evaluation corpus differs from training provenance")
    if (digest(metadata_path) if metadata_path else None) != provenance["metadata_sha256"]:
        raise ValueError("Evaluation metadata differs from training metadata")
    store = FeatureStore(resources, cache, allow_synthetic)
    if store.manifest != provenance["feature_manifest"] or digest(Path(resources)/"sequences.json") != provenance["sequence_sha256"]:
        raise ValueError("Evaluation features/reference differ from training")
    d = read_corpus(corpus)
    if feature_fingerprints(store, d.protein_reference_id.unique()) != provenance["feature_files"]:
        raise ValueError("Evaluation feature/structure file contents changed since training")
    fold, config = provenance["fold"], provenance["config"]
    transform = TargetTransform(state["transform"])
    metadata = Metadata(metadata_path, state["metadata_vocab"])
    cal = predict(model, loader(d[d.gene.isin(fold["calibration"])], store, transform, metadata, config, workers=workers), device)
    calibration = calibration_quantiles(cal)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "calibration.json", calibration)
    cal.to_csv(out / "calibration_predictions.csv", index=False)
    cal_metrics, cal_summary = metrics(cal)
    cal_metrics.to_csv(out / "calibration_metrics.csv", index=False)
    save_json(out / "calibration_summary.json", cal_summary)
    test = predict(model, loader(d[d.gene.isin(fold["test"])], store, transform, metadata, config, workers=workers), device)
    test = add_intervals(test, calibration)
    # Inverse transform uses training statistics only; unseen-assay scale transfer remains an assumption.
    test["prediction_score_units"] = [r.mu*transform.state.get(r.task, {"scale": 1.})["scale"]+transform.state.get(r.task, {"center": 0.})["center"] for r in test.itertuples()]
    scales = test.task.map({task: value["scale"] for task, value in transform.state.items()}).fillna(1.)
    centers = test.task.map({task: value["center"] for task, value in transform.state.items()}).fillna(0.)
    test["sigma_score_units"] = test.sigma*scales
    test["interval_low_score_units"] = test.interval_low*scales+centers
    test["interval_high_score_units"] = test.interval_high*scales+centers
    test.to_csv(out / "test_predictions.csv", index=False)
    per_assay, summary = metrics(test)
    summary["calibration_alpha"] = calibration["alpha"]
    summary["scale_transfer_limitation"] = "Training-only task scales are transferred to unseen assays; absolute scores and intervals require separate validation."
    per_assay.to_csv(out / "test_metrics.csv", index=False)
    save_json(out / "test_summary.json", summary)
    return summary
