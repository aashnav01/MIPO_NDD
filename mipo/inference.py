"""Checkpoint inference for requested substitutions on validated reference proteins."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .common import AA, TASKS, digest, to_device
from .data import FeatureStore, Metadata, TargetTransform, VariantDataset, collate
from .train import restore


def infer(variants, resources, cache, checkpoint, out, metadata_path=None, export_fields=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, state = restore(checkpoint, device)
    model.eval()
    config = state["provenance"]["config"]
    store = FeatureStore(resources, cache)
    if store.manifest != state["provenance"]["feature_manifest"]:
        raise ValueError("Inference feature model/window configuration differs from training")
    rows = pd.read_csv(variants, keep_default_na=False)
    required = {"gene", "wt", "ref_pos", "mut", "task"}
    if not required <= set(rows):
        raise ValueError(f"Variant input requires {sorted(required)}; optional assay_key for semantic metadata")
    if (~rows.task.isin(TASKS)).any() or (~rows.wt.isin(list(AA))).any() or (~rows.mut.isin(list(AA))).any():
        raise ValueError("Unknown task or amino acid")
    rows["ref_pos"] = pd.to_numeric(rows.ref_pos, errors="raise")
    if ((rows.ref_pos < 1) | (rows.ref_pos % 1 != 0)).any():
        raise ValueError("ref_pos must be a positive integer")
    rows.ref_pos = rows.ref_pos.astype(int)
    if "protein_reference_id" not in rows:
        refs = {}
        for reference_id, record in store.records.items():
            refs.setdefault(record.get("gene", reference_id), []).append(reference_id)
        ambiguous = [gene for gene in rows.gene.unique() if len(refs.get(gene, [])) != 1]
        if ambiguous:
            raise ValueError(f"Provide protein_reference_id for genes with multiple/missing constructs: {ambiguous}")
        rows["protein_reference_id"] = rows.gene.map({gene: ids[0] for gene, ids in refs.items()})
    rows["row_id"] = np.arange(len(rows))
    rows["variant_key"] = rows.protein_reference_id+":"+rows.wt+rows.ref_pos.astype(str)+rows.mut
    if "assay_key" not in rows:
        rows["assay_key"] = rows.gene+"::query"
    rows["score_value"] = 0.  # Dataset API placeholder; never passed into the model as input.
    transform = TargetTransform(state["transform"])
    ds = VariantDataset(rows, store, transform, Metadata(metadata_path, state["metadata_vocab"]), config)
    batches = DataLoader(ds, batch_size=config["batch_size"], collate_fn=collate)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    seen = {task for task, supported in zip(TASKS, model.seen_tasks.cpu().tolist()) if supported}
    with torch.inference_mode():
        for batch in batches:
            batch = to_device(batch, device)
            pred = model(batch)
            for j, row_id in enumerate(batch["row_id"].tolist()):
                row = rows.iloc[row_id]
                result = {k: row[k] for k in ["gene", "protein_reference_id", "wt", "ref_pos", "mut", "task", "assay_key"]}
                scale = transform.state.get(row.task, {"scale": 1., "center": 0.})
                result.update({"mu_standardized": float(pred["mu"][j]), "sigma_standardized": float(pred["sigma"][j]),
                               "score_prediction": float(pred["mu"][j])*scale["scale"]+scale["center"],
                               "task_supported": row.task in seen, "warning": "" if row.task in seen else "UNSUPERVISED_TASK_EXPLORATORY_ONLY"})
                results.append(result)
                if export_fields:
                    mask = batch["node_mask"][j].cpu().numpy()
                    conformers = batch["conformer_mask"][j].cpu().numpy()
                    np.savez_compressed(out / f"field_{row_id:06d}.npz",
                                        reference_positions=batch["ids"][j].cpu().numpy()[mask]+1,
                                        scalar_field=pred["field"][j].cpu().numpy()[mask],
                                        vector_field=pred["vector"][j].cpu().numpy()[conformers][:, mask],
                                        latent=pred["latent"][j].cpu().numpy(),
                                        modality_gates=pred["gates"][j].cpu().numpy())
    pd.DataFrame(results).to_csv(out / "predictions.csv", index=False)
