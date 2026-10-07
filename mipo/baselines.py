from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import save_json
from .corpus import read_corpus


def llr_baselines(corpus, out, resources=None, cache=None):
    """Unsupervised signed and absolute LLR correlation; never tunes sign on test."""
    d = read_corpus(corpus)
    if bool(resources) != bool(cache):
        raise ValueError("Provide both --resources and --cache")
    columns = ["llr_650M", "llr_150M"]
    if cache:
        from .data import FeatureStore
        from .common import AA
        store = FeatureStore(resources, cache)
        values = []
        for r in d.itertuples():
            llr = store.protein(r.protein_reference_id)["llr"]
            values.append(float(llr[r.ref_pos-1, AA.index(r.mut)]) if llr is not None else np.nan)
        d["llr_cached"] = values
        columns.append("llr_cached")
    records = []
    for (gene, assay, task), rows in d.groupby(["gene", "assay_key", "task"]):
        for column in columns:
            if column not in rows:
                continue
            score = pd.to_numeric(rows[column], errors="coerce")
            valid = score.notna()
            rho = float(spearmanr(rows.loc[valid, "score_value"], score[valid]).statistic) if valid.sum() >= 3 and score[valid].nunique() > 1 else np.nan
            records.append({"gene": gene, "assay_key": assay, "task": task, "model": column,
                            "available_rows": int(valid.sum()), "total_rows": len(rows), "signed_spearman": rho})
    result = pd.DataFrame(records)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "existing_llr_baselines.csv", index=False)
    save_json(out / "baseline_notes.json", {"interpretation": "Signed correlation against as-supplied targets. Score orientation and original LLR provenance are not inferred from the held-out gene.",
                                            "limitation": "Existing LLR columns cover Tier A only; run cached-LLR evaluation for an identical feature-generation protocol."})
    return result
