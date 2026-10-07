"""Structure-feature probe: does real geometry carry stability signal, and does it add to ESM?

This is the "explicit structure-feature probe" disjunct of the Tier 2 gate, in its
simplest honest form. It uses ONE geometric feature, the C-alpha contact number at the
variant's own site (count of other C-alpha atoms within R angstrom), because that is all
the model's graph can see: `VariantDataset.graph` reads only CA coordinates.

Pre-specified, so nothing here is tuned:
  * feature: contact number at R = 10 A, first conformer, covered residues only
    (R = 8 and 12 are sensitivity checks, not alternatives to pick from);
  * baseline: substitution identity (one-hot wt, one-hot mut);
  * optional ESM feature: the masked-marginal LLR of the substituted residue, read from a
    `mipo features` cache for the teachers and from the 1.5a scores file for the benchmark;
  * two fixed probe models, ridge(alpha=10) and HistGradientBoosting with library
    defaults; no search over any hyperparameter;
  * metric: pooled Spearman on per-protein z-scored values, via the SAME
    `mipo.fireprot.pooled_metrics` the 1.5a benchmark used, so numbers are comparable.

Roles are respected. Fits use S8754 TEACHER rows only. FireProt is BENCHMARK ONLY: it is
scored, never fitted. Teachers that are domain-level relatives of a benchmark protein
must already be removed (dedupe_s8754_homology.py); a probe run on un-deduped teachers
reports inflated transfer, which this project measured: removing 61 homologous teachers
cut the best transfer arm from 0.447 to 0.361.

Sign convention throughout: ddG > 0 = destabilizing, the project convention and the sign
S8754 was flipped to. The scorer handed to pooled_metrics negates its column (higher =
more stable), matching how 1.5a scored, so ddG-direction predictions pass through as they
are and the stability-high columns (LLR, mu) are pre-negated.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

R_MAIN, R_SENS = 10.0, (8.0, 12.0)
AA = "ACDEFGHIKLMNPQRSTVWY"


def contact_numbers(coords, conf, radius):
    """Per-residue CA contact number from the first conformer; NaN where uncovered."""
    xyz, ok = coords[0], conf[0] > 0.5
    out = np.full(len(xyz), np.nan)
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return out
    pts = xyz[idx]
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    out[idx] = (d < radius).sum(1) - 1
    return out


def attach_contacts(rows, sequences, structure_dir, radii):
    cache, cols = {}, {r: np.full(len(rows), np.nan) for r in radii}
    for i, r in enumerate(rows.itertuples()):
        key = sequences[r.protein_reference_id]["key"]
        if key not in cache:
            path = Path(structure_dir) / f"{key}.npz"
            if not path.exists():
                cache[key] = None
            else:
                with np.load(path, allow_pickle=False) as z:
                    cache[key] = {rad: contact_numbers(z["coords"], z["confidence"], rad) for rad in radii}
        if cache[key] is not None:
            for rad in radii:
                cols[rad][i] = cache[key][rad][r.ref_pos - 1]
    for rad in radii:
        rows[f"contact{int(rad)}"] = cols[rad]
    return rows


def attach_llr(rows, sequences, cache_dir):
    """Masked-marginal LLR (mutant minus wild-type logit) from a `mipo features` cache."""
    out = np.full(len(rows), np.nan)
    cache = {}
    for i, r in enumerate(rows.itertuples()):
        key = sequences[r.protein_reference_id]["key"]
        if key not in cache:
            path = Path(cache_dir) / key / "llr.npy"
            cache[key] = np.load(path) if path.exists() else None
        if cache[key] is not None:
            out[i] = cache[key][r.ref_pos - 1, AA.index(r.mut)]
    rows["llr"] = out
    return rows


def design(df, features, stats):
    """One-hot wt + one-hot mut, plus standardised continuous features."""
    X = np.zeros((len(df), 40 + len(features)))
    X[np.arange(len(df)), df.wt.map(AA.index).to_numpy()] = 1
    X[np.arange(len(df)), 20 + df.mut.map(AA.index).to_numpy()] = 1
    for k, f in enumerate(features):
        mean, std = stats[f]
        X[:, 40 + k] = (df[f].to_numpy() - mean) / std
    return X


def fit_predict(train, test, features, kind):
    stats = {f: (train[f].mean(), train[f].std(ddof=0) or 1.0) for f in features}
    Xtr, Xte = design(train, features, stats), design(test, features, stats)
    if kind == "ridge":
        model = Ridge(alpha=10.0)
    else:
        categorical = np.zeros(Xtr.shape[1], bool)
        categorical[:40] = True
        model = HistGradientBoostingRegressor(categorical_features=categorical, random_state=0)
    model.fit(Xtr, train.ddG.to_numpy())
    return model.predict(Xte)


def make_scorer(pooled_metrics):
    def score(df, column):
        """Pooled z-Spearman exactly as 1.5a computed it (the scorer negates its column)."""
        m = pooled_metrics(df[["gene", "ddG"]].assign(score=-df[column].to_numpy()), ["score"])
        return m["score"]["pooled_z_spearman"], m["_pooled_groups"]
    return score


def cluster_bootstrap(df, columns, score, reps, seed=0):
    """Resample whole proteins; returns one array of draws per column."""
    rng = np.random.default_rng(seed)
    genes = df.gene.unique()
    parts = {g: d for g, d in df.groupby("gene")}
    draws = {c: [] for c in columns}
    for _ in range(reps):
        pick = rng.choice(genes, len(genes), replace=True)
        boot = pd.concat([parts[g].assign(gene=f"{g}#{k}") for k, g in enumerate(pick)], ignore_index=True)
        for c in columns:
            draws[c].append(score(boot, c)[0])
    return {c: np.array(v) for c, v in draws.items()}


def zs(d, col):
    s = d[col].std(ddof=0)
    return None if not s > 0 else (d[col] - d[col].mean()) / s


def pooled_z(df, x, y):
    pairs = [(zs(d, x), zs(d, y)) for _, d in df.groupby("gene") if len(d) >= 2]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    return float(spearmanr(np.concatenate([a for a, _ in pairs]), np.concatenate([b for _, b in pairs])).statistic)


def macro_spearman(df, x, y, min_n=8):
    vals = [spearmanr(d[x], d[y]).statistic for _, d in df.groupby("gene")
            if len(d) >= min_n and d[x].nunique() > 1 and d[y].nunique() > 1]
    return float(np.mean(vals)), len(vals)


def leave_protein_out_expectation(df):
    """E[ddG | wt->mut] from OTHER proteins only; falls back to wt, then global mean."""
    pair = df.groupby(["wt", "mut"]).ddG.agg(["sum", "count"])
    wt = df.groupby("wt").ddG.agg(["sum", "count"])
    own_pair = df.groupby(["gene", "wt", "mut"]).ddG.agg(["sum", "count"])
    own_wt = df.groupby(["gene", "wt"]).ddG.agg(["sum", "count"])
    gene_sum, gene_n = df.groupby("gene").ddG.sum(), df.groupby("gene").size()
    total, n = df.ddG.sum(), len(df)
    out = np.empty(len(df))
    for i, r in enumerate(df.itertuples()):
        ps, pc = pair.loc[(r.wt, r.mut)]
        os_, oc = own_pair.loc[(r.gene, r.wt, r.mut)]
        if pc - oc >= 5:
            out[i] = (ps - os_) / (pc - oc)
            continue
        ws, wc = wt.loc[r.wt]
        ows, owc = own_wt.loc[(r.gene, r.wt)]
        out[i] = (ws - ows) / (wc - owc) if wc - owc >= 5 else (total - gene_sum[r.gene]) / (n - gene_n[r.gene])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher-rows", required=True, help="s8754_corpus_rows.csv")
    p.add_argument("--teacher-sequences", required=True)
    p.add_argument("--teacher-structures", required=True, help="Resources dir containing structures/")
    p.add_argument("--benchmark-corpus", required=True)
    p.add_argument("--benchmark-sequences", required=True)
    p.add_argument("--benchmark-structures", required=True, help="Directory holding the benchmark .npz files")
    p.add_argument("--teacher-cache", help="`mipo features` cache for the teachers; enables the LLR arms")
    p.add_argument("--benchmark-scores", help="1.5a fireprot_scores.csv (llr and trained-mu columns)")
    p.add_argument("--mipo-src", default=".", help="Directory containing the mipo package (for pooled_metrics)")
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    sys.path.insert(0, a.mipo_src)
    from mipo.fireprot import pooled_metrics  # noqa: E402
    score = make_scorer(pooled_metrics)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {}

    teacher = pd.read_csv(a.teacher_rows).rename(columns={"score_value": "ddG"})
    tseq = json.loads(Path(a.teacher_sequences).read_text())
    report = pd.read_csv(Path(a.teacher_structures) / "structures/s8754_structure_report.csv").set_index("gene")
    teacher["source"] = teacher.gene.map(report.source)
    teacher = attach_contacts(teacher, tseq, Path(a.teacher_structures) / "structures", [R_MAIN, *R_SENS])
    use_llr = bool(a.teacher_cache and a.benchmark_scores)
    if use_llr:
        teacher = attach_llr(teacher, tseq, a.teacher_cache)
    have = teacher.contact10.notna() & (teacher.llr.notna() if use_llr else True)
    print(f"teachers: {len(teacher)} variants, {teacher.gene.nunique()} proteins; usable {int(have.sum())}")
    exp = teacher[have & (teacher.source == "rcsb")].copy()
    print(f"experimental structures: {len(exp)} variants, {exp.gene.nunique()} proteins "
          f"(AlphaFold teachers are excluded from every fit and statistic below)\n")

    print("=" * 76 + "\nA. DESCRIPTIVE on experimental structures (nothing fitted)\n" + "=" * 76)
    desc = {}
    for rad in (R_MAIN, *R_SENS):
        col = f"contact{int(rad)}"
        pooled, (macro, n) = pooled_z(exp, col, "ddG"), macro_spearman(exp, col, "ddG")
        desc[col] = {"pooled_z_spearman": pooled, "macro_within_protein": macro, "proteins": n}
        print(f"  R={rad:>4.0f} A  pooled z-Spearman(contact, ddG) {pooled:+.3f} | "
              f"macro within-protein {macro:+.3f} ({n} proteins)")
    exp["bin"] = pd.qcut(exp.contact10, 3, labels=["exposed", "intermediate", "buried"])
    tab = exp.groupby("bin", observed=True).ddG.agg(["count", "mean", "median"])
    tab["destabilizing"] = exp.groupby("bin", observed=True).ddG.apply(lambda s: (s > 0).mean())
    print("  ddG by burial tercile:\n" + tab.round(2).to_string())
    exp["resid"] = exp.ddG - leave_protein_out_expectation(exp)
    adj = pooled_z(exp, "contact10", "resid"), macro_spearman(exp, "contact10", "resid")[0]
    desc["after_substitution_adjustment"] = {"pooled_z_spearman": adj[0], "macro_within_protein": adj[1]}
    print(f"  after removing the leave-protein-out wt->mut expectation: pooled {adj[0]:+.3f} | macro {adj[1]:+.3f}")
    if use_llr:
        # LLR is stability-high, so a good LLR correlates NEGATIVELY with ddG.
        desc["llr_vs_ddG_pooled_z"] = pooled_z(exp, "llr", "ddG")
        desc["contact_vs_llr_pooled_z"] = pooled_z(exp, "contact10", "llr")
        print(f"  ESM LLR vs ddG pooled z-Spearman {desc['llr_vs_ddG_pooled_z']:+.3f} (negative = LLR is predictive); "
              f"contact vs LLR {desc['contact_vs_llr_pooled_z']:+.3f}")
    result["descriptive"] = desc

    print("\n" + "=" * 76 + "\nB. TRANSFER: fit on teachers, score FireProt (benchmark, never fitted)\n" + "=" * 76)
    fp = pd.read_csv(a.benchmark_corpus, keep_default_na=False)
    fp["row_id"] = np.arange(len(fp))
    fp = fp.rename(columns={"score_value": "ddG"})
    fseq = json.loads(Path(a.benchmark_sequences).read_text())
    fp = attach_contacts(fp, fseq, a.benchmark_structures, [R_MAIN])
    if use_llr:
        scores = pd.read_csv(a.benchmark_scores)
        fp = fp.merge(scores[["row_id", "llr", "mu"]], on="row_id", validate="one_to_one")
    covered = fp.contact10.notna()
    print(f"benchmark: {len(fp)} rows; {int(covered.sum())} with a contact value "
          f"({fp[covered].gene.nunique()} proteins). Rows without one are excluded from ALL arms equally.")
    fp = fp[covered].copy()

    arm_features = {"substitution only": [], "substitution + contact": ["contact10"]}
    if use_llr:
        arm_features.update({"substitution + LLR": ["llr"], "substitution + LLR + contact": ["llr", "contact10"]})
    preds = {}
    for kind in ("ridge", "hgb"):
        for name, feats in arm_features.items():
            preds[f"{kind}: {name}"] = fit_predict(exp, fp, feats, kind)  # ddG-direction predictions
    preds["contact only (no fit)"] = fp.contact10.to_numpy().astype(float)  # more contacts, more destabilizing
    if use_llr:
        preds["zero-shot ESM LLR (no fit)"] = -fp.llr.to_numpy().astype(float)  # stability-high, pre-negated
        preds["trained esm_mlp mu (1.5a)"] = -fp.mu.to_numpy().astype(float)
    cols = {}
    for name, pr in preds.items():
        cols[name] = f"p{len(cols)}"
        fp[cols[name]] = pr

    no1stn = fp.gene != "FP_1STN_1EY0"
    draws = cluster_bootstrap(fp, list(cols.values()), score, a.bootstrap)
    print(f"\n  pooled z-Spearman vs ddG; 95% CI = cluster bootstrap over proteins ({a.bootstrap} resamples)")
    print(f"  {'arm':<34s}{'all':>8s}{'w/o 1STN':>10s}{'groups':>8s}   95% CI")
    table = {}
    for name, col in cols.items():
        rho, k = score(fp, col)
        rho2, _ = score(fp[no1stn], col)
        lo, hi = np.percentile(draws[col], [2.5, 97.5])
        table[name] = {"all": rho, "without_1STN": rho2, "groups": k, "ci95": [float(lo), float(hi)]}
        print(f"  {name:<34s}{rho:>+8.3f}{rho2:>+10.3f}{k:>8d}   [{lo:+.3f}, {hi:+.3f}]")
    result["transfer"] = table

    print("\n  PAIRED effect of adding the contact feature (the gate-relevant numbers):")
    effects = {}
    pairs = [("substitution + contact", "substitution only", "contact beyond substitution identity")]
    if use_llr:
        pairs.append(("substitution + LLR + contact", "substitution + LLR", "contact beyond ESM LLR"))
    for with_name, base_name, label in pairs:
        for kind in ("ridge", "hgb"):
            diff = draws[cols[f"{kind}: {with_name}"]] - draws[cols[f"{kind}: {base_name}"]]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            effects[f"{label} [{kind}]"] = {"mean": float(diff.mean()), "ci95": [float(lo), float(hi)],
                                            "share_positive": float((diff > 0).mean())}
            print(f"  {label:<38s}[{kind:5s}] {diff.mean():+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  "
                  f"positive in {(diff > 0).mean():.0%}")
    result["effects"] = effects

    (out / "structure_probe.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    fp.drop(columns=list(cols.values())).to_csv(out / "probe_benchmark_rows.csv", index=False)
    exp.to_csv(out / "probe_teacher_rows.csv", index=False)
    print(f"\nwrote {out}/structure_probe.json")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
