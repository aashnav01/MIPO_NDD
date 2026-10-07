"""Combine seeds for the SAME outer fold; recalibrate on their calibration predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mipo.common import save_json
from mipo.metrics import add_intervals, calibration_quantiles, metrics


def combine(runs, filename):
    frames = [pd.read_csv(Path(run)/filename).sort_values("row_id").reset_index(drop=True) for run in runs]
    first = frames[0].copy()
    for other in frames[1:]:
        if not first[["row_id", "variant_key", "assay_key", "y", "task_seen_in_training"]].equals(other[["row_id", "variant_key", "assay_key", "y", "task_seen_in_training"]]):
            raise ValueError("Ensemble rows/targets/support differ; use seeds of the same fold and preprocessing")
    means = np.stack([x.mu.to_numpy() for x in frames])
    variances = np.stack([x.sigma.to_numpy()**2 for x in frames])
    first["mu"] = means.mean(0)
    first["aleatoric_variance"] = variances.mean(0)
    first["epistemic_variance"] = means.var(0)
    first["sigma"] = np.sqrt(first.aleatoric_variance+first.epistemic_variance)
    first = first.drop(columns=[x for x in ["interval_low", "interval_high", "prediction_score_units"] if x in first])
    return first


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if len(a.runs) < 2:
        raise ValueError("At least two seed runs required")
    provenance = [json.loads((Path(r)/"provenance.json").read_text()) for r in a.runs]
    for other in provenance[1:]:
        for key in ["fold", "corpus_sha256", "split_sha256", "feature_files", "metadata_sha256"]:
            if other[key] != provenance[0][key]:
                raise ValueError(f"Ensemble provenance mismatch: {key}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cal = combine(a.runs, "calibration_predictions.csv")
    calibration = calibration_quantiles(cal)
    test = add_intervals(combine(a.runs, "test_predictions.csv"), calibration)
    test.to_csv(out/"test_predictions.csv", index=False)
    cal.to_csv(out/"calibration_predictions.csv", index=False)
    per_assay, summary = metrics(test)
    per_assay.to_csv(out/"test_metrics.csv", index=False)
    save_json(out/"test_summary.json", summary)
    save_json(out/"calibration.json", calibration)


if __name__ == "__main__":
    main()
