"""Judge the 1.5b run against the three bars fixed before training.

  T1  1.5b vs v1, same model and fold: did stage-1 stability supervision move `mu`?   (the 1.5b claim)
  T2  1.5b vs the probe ceilings: does the network beat a 42-parameter ridge on ESM LLR?
  T3  mipo minus esm_mlp in 1.5b, against the same contrast in v1: does the structure leg move now
      that it has structured stability supervision?                                     (the Tier 2 question)

Only T1 is 1.5b's own claim. T2 and T3 are diagnostics and are reported whatever they say.

Everything is the pooled z-Spearman that 1.5a used (`mipo.fireprot.pooled_metrics`), on two row sets:
all proteins (the 1.5a number) and the proteins with real coordinates (the set the probe was scored on,
which is where the ceilings below were measured). Uncertainty is a cluster bootstrap over proteins, and
every difference is PAIRED: the same resampled proteins feed both sides.

Ceilings are measured values from STRUCTURE_PROBE_RESULTS.md section 6, held constant here:
ridge(substitution + ESM LLR) 0.451 and ridge(substitution + ESM LLR + contact) 0.474.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CEILING_LLR, CEILING_LLR_CONTACT = 0.451, 0.474


def load(root, tag):
    path = Path(root) / tag / "fireprot_scores.csv"
    return pd.read_csv(path) if path.exists() else None


def pooled(pooled_metrics, df, column="mu"):
    m = pooled_metrics(df, [column])
    return m[column]["pooled_z_spearman"]


def boot(pooled_metrics, frames, reps, seed=0):
    """Draw protein resamples once and score every frame on the SAME draws, so differences are paired."""
    rng = np.random.default_rng(seed)
    genes = sorted(set.intersection(*(set(f.gene) for f in frames.values())))
    parts = {k: {g: d for g, d in f.groupby("gene")} for k, f in frames.items()}
    draws = {k: [] for k in frames}
    for _ in range(reps):
        pick = rng.choice(genes, len(genes), replace=True)
        for k in frames:
            boot_df = pd.concat([parts[k][g].assign(gene=f"{g}#{i}") for i, g in enumerate(pick)], ignore_index=True)
            draws[k].append(pooled(pooled_metrics, boot_df))
    return {k: np.array(v) for k, v in draws.items()}


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return f"[{lo:+.3f}, {hi:+.3f}]"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v1", required=True, help="Folder of 1.5a results (runs_<model>_seed_42_fold_<n>/)")
    p.add_argument("--new", required=True, help="Folder of 1.5b results, same naming")
    p.add_argument("--subset-rows", required=True, help="probe_benchmark_rows.csv: rows with real coordinates")
    p.add_argument("--folds", default="12,57")
    p.add_argument("--mipo-src", default=".")
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    sys.path.insert(0, a.mipo_src)
    from mipo.fireprot import pooled_metrics

    subset = set(pd.read_csv(a.subset_rows).row_id)
    folds = [int(f) for f in a.folds.split(",")]
    sets = {"all proteins (1.5a basis)": None, "proteins with coordinates (probe basis)": subset}
    result = {}
    for label, rows in sets.items():
        print("=" * 78 + f"\n{label}\n" + "=" * 78)
        block = {}
        data = {}
        for model in ("mipo", "esm_mlp"):
            for fold in folds:
                tag = f"runs_{model}_seed_42_fold_{fold}"
                for name, root in (("v1", a.v1), ("new", a.new)):
                    df = load(root, tag)
                    if df is None:
                        print(f"  MISSING {name}/{tag}")
                        continue
                    data[(name, model, fold)] = df if rows is None else df[df.row_id.isin(rows)]
        if not data:
            continue
        print(f"  {'run':<22s}{'v1':>9s}{'1.5b':>9s}{'  1.5b - v1 (paired)':>26s}   vs ceilings (1.5b)")
        # T1 and T2
        for model in ("mipo", "esm_mlp"):
            for fold in folds:
                k1, k2 = ("v1", model, fold), ("new", model, fold)
                if k1 not in data or k2 not in data:
                    continue
                draws = boot(pooled_metrics, {"v1": data[k1], "new": data[k2]}, a.bootstrap)
                v1v, nv = pooled(pooled_metrics, data[k1]), pooled(pooled_metrics, data[k2])
                d = draws["new"] - draws["v1"]
                above = np.mean(draws["new"] > CEILING_LLR_CONTACT)
                block[f"T1 {model} fold {fold}"] = {"v1": v1v, "new": nv, "delta": nv - v1v, "delta_ci95": list(np.percentile(d, [2.5, 97.5]))}
                block[f"T2 {model} fold {fold}"] = {"new": nv, "ci95": list(np.percentile(draws["new"], [2.5, 97.5])),
                                                    "vs_ridge_llr": nv - CEILING_LLR, "vs_ridge_llr_contact": nv - CEILING_LLR_CONTACT,
                                                    "share_of_resamples_above_0.474": float(above)}
                print(f"  {model + ' fold ' + str(fold):<22s}{v1v:>+9.3f}{nv:>+9.3f}   {nv - v1v:>+8.3f} {ci(d):<17s}"
                      f"   {nv - CEILING_LLR:+.3f} / {nv - CEILING_LLR_CONTACT:+.3f}  (above 0.474 in {above:.0%})")
        # T3
        print(f"\n  T3  mipo - esm_mlp (the structure leg), per fold")
        for fold in folds:
            need = [("new", "mipo", fold), ("new", "esm_mlp", fold), ("v1", "mipo", fold), ("v1", "esm_mlp", fold)]
            if any(k not in data for k in need):
                continue
            frames = {"nm": data[need[0]], "ne": data[need[1]], "vm": data[need[2]], "ve": data[need[3]]}
            draws = boot(pooled_metrics, frames, a.bootstrap)
            new_c, v1_c = draws["nm"] - draws["ne"], draws["vm"] - draws["ve"]
            did = new_c - v1_c
            block[f"T3 fold {fold}"] = {"new_contrast": float(np.mean(new_c)), "new_ci95": list(np.percentile(new_c, [2.5, 97.5])),
                                       "v1_contrast": float(np.mean(v1_c)), "v1_ci95": list(np.percentile(v1_c, [2.5, 97.5])),
                                       "difference_in_differences": float(np.mean(did)), "did_ci95": list(np.percentile(did, [2.5, 97.5]))}
            print(f"      fold {fold}: 1.5b {np.mean(new_c):+.3f} {ci(new_c)} | v1 {np.mean(v1_c):+.3f} {ci(v1_c)} | "
                  f"change {np.mean(did):+.3f} {ci(did)}")
        result[label] = block
        print()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare_15b_vs_v1.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"wrote {out}/compare_15b_vs_v1.json")


if __name__ == "__main__":
    main()
