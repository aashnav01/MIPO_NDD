import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def metrics(predictions):
    if predictions.empty:
        raise ValueError("Cannot evaluate empty predictions")
    if not np.isfinite(predictions[["y", "mu", "sigma"]].to_numpy()).all() or (predictions.sigma <= 0).any():
        raise ValueError("Metrics require finite targets/predictions and positive sigma")
    reports = []
    for (gene, assay, task), d in predictions.groupby(["gene", "assay_key", "task"]):
        y, mu, sigma = d.y.to_numpy(), d.mu.to_numpy(), d.sigma.to_numpy()
        valid = len(y) >= 3 and np.ptp(y) > 0 and np.ptp(mu) > 0
        reports.append({"gene": gene, "assay_key": assay, "task": task, "n": len(d),
                        "spearman": float(spearmanr(y, mu).statistic) if valid else np.nan,
                        "pearson": float(pearsonr(y, mu).statistic) if valid else np.nan,
                        "rmse": float(np.sqrt(np.mean((y-mu)**2))),
                        "mae": float(np.mean(np.abs(y-mu))),
                        "gaussian_nll": float(np.mean(.5*((y-mu)/sigma)**2+np.log(sigma)+.5*np.log(2*np.pi))),
                        "coverage_90_gaussian": float(np.mean(np.abs(y-mu) <= 1.644854*sigma)),
                        "bias": float(np.mean(mu-y)),
                        "target_mean": float(np.mean(y)), "prediction_mean": float(np.mean(mu)),
                        "target_std": float(np.std(y)), "prediction_std": float(np.std(mu)),
                        "mean_sigma": float(np.mean(sigma)),
                        "task_seen_in_training": bool(d.task_seen_in_training.iloc[0])})
        if {"interval_low", "interval_high"} <= set(d):
            available = d.interval_low.notna() & d.interval_high.notna()
            reports[-1].update({
                "calibrated_interval_rows": int(available.sum()),
                "coverage_calibrated": float(((d.y[available] >= d.interval_low[available]) &
                                               (d.y[available] <= d.interval_high[available])).mean()) if available.any() else np.nan,
                "mean_calibrated_width": float((d.interval_high[available]-d.interval_low[available]).mean()) if available.any() else np.nan})
    assays = pd.DataFrame(reports)
    gene_scores = assays.groupby("gene").spearman.mean()
    summary = {"macro_gene_spearman": float(gene_scores.mean()) if gene_scores.notna().any() else None,
               "genes": len(gene_scores), "genes_with_defined_spearman": int(gene_scores.notna().sum()),
               "assays": len(assays), "unsupported_task_rows": int((~predictions.task_seen_in_training).sum()),
               "metric_units": "training_task_transformed_score",
               "per_gene": {gene: float(value) if pd.notna(value) else None for gene, value in gene_scores.items()},
               "per_task": {task: {"macro_gene_spearman": float(part.groupby('gene').spearman.mean().mean())
                                   if part.spearman.notna().any() else None, "genes": part.gene.nunique(),
                                   "macro_gene_coverage_90_gaussian": float(part.groupby('gene').coverage_90_gaussian.mean().mean()),
                                   "macro_gene_gaussian_nll": float(part.groupby('gene').gaussian_nll.mean().mean())}
                            for task, part in assays.groupby("task")}}
    if "calibrated_interval_rows" in assays:
        summary["calibrated_interval_rows"] = int(assays.calibrated_interval_rows.sum())
        summary["rows_without_calibrated_intervals"] = int(len(predictions)-assays.calibrated_interval_rows.sum())
        for task, part in assays.groupby("task"):
            value = part.groupby("gene").coverage_calibrated.mean().mean()
            summary["per_task"][task]["macro_gene_coverage_calibrated"] = float(value) if pd.notna(value) else None
    return assays, summary


def calibration_quantiles(predictions, alpha=.1):
    result = {"alpha": alpha, "method": "empirical_split_normalized_residual", "tasks": {},
              "limitation": "Variant exchangeability across new genes is not assured; these are empirical intervals, not guaranteed gene-OOD coverage."}
    for task, d in predictions.groupby("task"):
        if len(d) < 30 or not d.task_seen_in_training.all():
            continue
        residual = np.abs(d.y-d.mu)/d.sigma
        rank = min(len(d)-1, int(np.ceil((len(d)+1)*(1-alpha)))-1)
        result["tasks"][task] = {"q": float(np.sort(residual)[rank]), "rows": len(d), "genes": d.gene.nunique()}
    return result


def add_intervals(predictions, calibration):
    d = predictions.copy()
    q = d.task.map({k: x["q"] for k, x in calibration["tasks"].items()})
    q = q.where(d.task_seen_in_training)
    d["interval_low"] = d.mu-q*d.sigma
    d["interval_high"] = d.mu+q*d.sigma
    return d


def bootstrap_genes(assays, seed=42, draws=2000):
    values = assays.groupby("gene").spearman.mean().dropna().to_numpy()
    if len(values) < 2:
        return {"mean": float(values[0]) if len(values) else None, "ci95": None, "n_genes": len(values)}
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, (draws, len(values)), replace=True), axis=1)
    return {"mean": float(values.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(), "n_genes": len(values)}
