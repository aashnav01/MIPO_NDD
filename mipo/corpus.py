"""Validate labels without manufacturing missing assay measurements."""
from pathlib import Path

import numpy as np
import pandas as pd

from .common import AA, TYPE_MAP, digest, save_json


def read_corpus(path):
    d = pd.read_parquet(path) if str(path).endswith('.parquet') else pd.read_csv(path, keep_default_na=False, low_memory=False)
    required = {"gene", "wt", "ref_pos", "mut", "score_value", "score_kind", "assay_id", "assay_type", "supervision_tier"}
    if required - set(d):
        raise ValueError(f"Missing columns: {sorted(required - set(d))}")
    for col in ["ref_pos", "score_value"]:
        d[col] = pd.to_numeric(d[col], errors="raise")
    if not np.isfinite(d[["ref_pos", "score_value"]].to_numpy()).all():
        raise ValueError("Nonfinite target or position")
    if ((d.ref_pos < 1) | (d.ref_pos % 1 != 0)).any():
        raise ValueError("ref_pos must be a positive 1-based integer")
    d.ref_pos = d.ref_pos.astype(int)
    if "protein_reference_id" not in d:
        d["protein_reference_id"] = d.gene
    if (d.protein_reference_id.str.len() == 0).any():
        raise ValueError("Empty protein reference ID")
    if (~d.wt.isin(list(AA)) | ~d.mut.isin(list(AA)) | (d.wt == d.mut)).any():
        raise ValueError("Only canonical single amino-acid substitutions are supported")
    if (d.gene.str.len() == 0).any() or (d.assay_id.str.len() == 0).any():
        raise ValueError("Empty gene or assay ID")
    d["task"] = d.assay_type.map(TYPE_MAP)
    d.loc[d.score_kind == "collapsed_consensus_y", "task"] = "consensus"
    if d.task.isna().any():
        raise ValueError(f"Unknown assay types: {d.loc[d.task.isna(), 'assay_type'].unique()}")
    d["assay_key"] = d.gene + "::" + d.assay_id
    d["variant_key"] = d.protein_reference_id + ":" + d.wt + d.ref_pos.astype(str) + d.mut
    if d.duplicated(["variant_key", "assay_key"]).any():
        raise ValueError("Duplicate variant/assay rows: resolve provenance before training")
    if (d.groupby(["protein_reference_id", "ref_pos"]).wt.nunique() > 1).any():
        raise ValueError("Conflicting WT residues at a gene position")
    if (d.groupby("protein_reference_id").gene.nunique() > 1).any():
        raise ValueError("One protein reference maps to several gene holdout groups")
    if (d.groupby("assay_key").task.nunique() > 1).any():
        raise ValueError("One assay key has multiple measurement types")
    d["row_id"] = np.arange(len(d))
    return d


def audit(path, out):
    d = read_corpus(path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    coverage = d.groupby(["task", "gene", "assay_id"]).size().rename("rows").reset_index()
    coverage.to_csv(out / "coverage.csv", index=False)
    report = {"source_sha256": digest(path), "rows": len(d), "genes": d.gene.nunique(),
              "assays": d.assay_key.nunique(), "tiers": d.supervision_tier.value_counts().to_dict(),
              "task_rows": d.task.value_counts().to_dict(),
              "task_genes": d.groupby("task").gene.nunique().to_dict(),
              "genes_with_multiple_tasks": int((d.groupby("gene").task.nunique() > 1).sum()),
              "existing_llr_missing": {c: int(pd.to_numeric(d[c], errors="coerce").isna().sum())
                                       for c in ["llr_650M", "llr_150M"] if c in d},
              "notes": ["train_target_0_1 is not used: its normalization provenance is unverified.",
                        "Consensus labels are not reconstructed activity/abundance labels.",
                        "A task present in one gene has no same-task supervision when that gene is held out."]}
    save_json(out / "audit.json", report)
    meta = d[["assay_key", "task", "score_kind"]].drop_duplicates().copy()
    for c in ["host", "selection", "system", "treatment", "region"]:
        meta[c] = "unknown"
    meta["orientation"] = "as_supplied"
    meta.to_csv(out / "assay_metadata.csv", index=False)
    return report
