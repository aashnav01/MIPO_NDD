"""Compare trained predictions and cached LLR on exactly the same held-out rows."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import AA


def matched_llr_comparison(predictions, corpus, store):
    keys = ["row_id", "gene", "assay_key", "variant_key", "task", "score_value"]
    aligned = predictions.merge(corpus[keys+["protein_reference_id", "ref_pos", "mut"]],
                                on=keys, how="inner", validate="one_to_one")
    if len(aligned) != len(predictions) or aligned.empty:
        raise ValueError("Prediction rows/targets do not exactly match the current corpus")
    llrs = []
    for row in aligned.itertuples():
        cache = store.protein(row.protein_reference_id)["llr"]
        llrs.append(float(cache[row.ref_pos-1, AA.index(row.mut)]) if cache is not None else np.nan)
    aligned["llr_cached"] = llrs
    if not np.isfinite(aligned[["score_value", "mu", "llr_cached"]].to_numpy()).all():
        raise ValueError("Matched comparison requires finite predictions and cached LLR for every test row")
    records = []
    for (gene, assay, task), d in aligned.groupby(["gene", "assay_key", "task"]):
        def rho(column):
            return float(spearmanr(d.score_value, d[column]).statistic) if len(d) >= 3 and d.score_value.nunique() > 1 and d[column].nunique() > 1 else np.nan
        trained, baseline = rho("mu"), rho("llr_cached")
        records.append({"gene": gene, "assay_key": assay, "task": task, "n": len(d),
                        "model_spearman": trained, "llr_spearman": baseline, "delta": trained-baseline})
    assays = pd.DataFrame(records)
    complete = bool(assays[["model_spearman", "llr_spearman"]].notna().all().all())
    gene_means = assays.groupby("gene")[["model_spearman", "llr_spearman", "delta"]].mean()
    summary = {"matched_rows": len(aligned), "assays": len(assays), "genes": len(gene_means),
               "all_correlations_defined": complete,
               "model_macro_gene_spearman": float(gene_means.model_spearman.mean()) if complete else None,
               "llr_macro_gene_spearman": float(gene_means.llr_spearman.mean()) if complete else None,
               "macro_gene_delta": float(gene_means.delta.mean()) if complete else None,
               "note": "Exact shared variants, signed LLR, assay mean within gene then equal gene mean. Undefined assays suppress the headline comparison; no test-based sign selection."}
    return assays, summary
