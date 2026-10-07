"""Post-hoc 1.5a reads: raw-sign direction accuracy, 1STN leave-out, per-protein spread.

Reads only fireprot_scores.csv (no rerun). Reproduces pooled_metrics for the
z-Spearman so the printed numbers are checkable against fireprot_metrics.json.
Sign convention as in mipo/fireprot.py: ddG > 0 = destabilizing, scorers run
higher = more stable, so the target is -ddG.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DOMINANT = "FP_1STN_1EY0"


def zscore(v):
    v = np.asarray(v, float)
    s = v.std(ddof=0)
    return None if not np.isfinite(s) or s == 0 else (v - v.mean()) / s


def pooled(df, col):
    """Pooled z-Spearman + z-sign direction accuracy, as in mipo.fireprot.pooled_metrics."""
    sc, tg = [], []
    for _, g in df.groupby("gene"):
        if len(g) < 2:
            continue
        tz = zscore(g.ddG.to_numpy())
        sz = zscore(g[col].to_numpy())
        if tz is None or sz is None:
            continue
        sc.append(sz)
        tg.append(-tz)
    if not sc:
        return float("nan"), float("nan"), 0
    s, t = np.concatenate(sc), np.concatenate(tg)
    rho = float(spearmanr(s, t).statistic)
    keep = (s != 0) & (t != 0)
    return rho, float(((s[keep] > 0) == (t[keep] > 0)).mean()), len(s)


def raw_sign(df, col):
    """Accuracy on the RAW sign of ddG: the quantity FoldX/Rosetta/ThermoMPNN papers report.

    Scorers are higher = more stable, so a positive score should mean ddG < 0.
    A scorer has no zero point across proteins, so it is centred per protein
    first; without that, the column's arbitrary offset decides the answer.
    Rows at exactly ddG == 0 are excluded (no true direction).
    """
    d = df[df.ddG != 0]
    centred = d[col] - d.groupby("gene")[col].transform("median")
    return float(((centred > 0) == (d.ddG < 0)).mean()), len(d)


def per_protein(df, col, min_n=5):
    rows = []
    for name, g in df.groupby("gene"):
        if len(g) < min_n:
            continue
        if zscore(g.ddG.to_numpy()) is None or zscore(g[col].to_numpy()) is None:
            continue
        rows.append((name, len(g), float(spearmanr(g[col], -g.ddG).statistic)))
    return pd.DataFrame(rows, columns=["gene", "n", "rho"])


root = Path(sys.argv[1])
runs = sorted(p.parent for p in root.rglob("fireprot_scores.csv"))
print(f"{'run':30s} {'scorer':6s} {'pooled_rho':>10s} {'z_dir':>7s} {'raw_dir':>8s} "
      f"{'rho_no1STN':>11s} {'raw_no1STN':>11s}")
summary = {}
for run in runs:
    df = pd.read_csv(run / "fireprot_scores.csv")
    assert df.ddG.notna().all() and len(df) == 2427, f"{run.name}: unexpected {len(df)} rows"
    out = df[df.gene != DOMINANT]
    for col in ["llr", "mu"]:
        rho, zdir, n = pooled(df, col)
        rdir, nraw = raw_sign(df, col)
        rho_o, _, _ = pooled(out, col)
        rdir_o, _ = raw_sign(out, col)
        summary[(run.name, col)] = (rho, zdir, rdir, rho_o, rdir_o)
        print(f"{run.name:30s} {col:6s} {rho:10.4f} {zdir:7.4f} {rdir:8.4f} "
              f"{rho_o:11.4f} {rdir_o:11.4f}")
print(f"\nrows: {len(df)} total, {len(out)} without {DOMINANT} "
      f"({len(df) - len(out)} rows = {(len(df) - len(out)) / len(df):.0%}), "
      f"pooled rows per metric {n}, raw-sign rows {nraw}")

print(f"\nper-protein rho (n>=5 rows), mu vs llr:")
for run in runs:
    df = pd.read_csv(run / "fireprot_scores.csv")
    pm, pl = per_protein(df, "mu"), per_protein(df, "llr")
    merged = pm.merge(pl, on=["gene", "n"], suffixes=("_mu", "_llr"))
    print(f"  {run.name:30s} proteins {len(merged):3d}  "
          f"mu median {merged.rho_mu.median():6.3f} IQR [{merged.rho_mu.quantile(.25):.3f}, "
          f"{merged.rho_mu.quantile(.75):.3f}]  beats llr in {int((merged.rho_mu > merged.rho_llr).sum())}/{len(merged)}")

mus = [summary[(r.name, "mu")][0] for r in runs]
print(f"\nmu pooled rho across the 4 checkpoints: {min(mus):.4f}-{max(mus):.4f} "
      f"(spread {max(mus) - min(mus):.4f}); llr floor {summary[(runs[0].name, 'llr')][0]:.4f}")

# Per-protein paired comparison of the two architectures, matched on fold.
for fold in ["fold_12", "fold_57"]:
    a = [r for r in runs if "esm_mlp" in r.name and fold in r.name]
    b = [r for r in runs if "mipo" in r.name and fold in r.name]
    if not (a and b):
        continue
    pa = per_protein(pd.read_csv(a[0] / "fireprot_scores.csv"), "mu")
    pb = per_protein(pd.read_csv(b[0] / "fireprot_scores.csv"), "mu")
    m = pa.merge(pb, on=["gene", "n"], suffixes=("_esm", "_mipo"))
    diff = m.rho_mipo - m.rho_esm
    print(f"{fold}: mipo - esm_mlp per-protein rho: median {diff.median():+.3f}, "
          f"mipo better in {int((diff > 0).sum())}/{len(m)} proteins")
