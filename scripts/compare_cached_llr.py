"""Generate exact-row cached-LLR comparisons for completed experiment runs."""
import argparse
import json
from pathlib import Path

import pandas as pd

from mipo.common import digest, save_json
from mipo.comparison import matched_llr_comparison
from mipo.corpus import read_corpus
from mipo.data import FeatureStore
from mipo.train import feature_fingerprints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["corpus", "resources", "cache", "runs"]:
        parser.add_argument("--"+key, required=True)
    args = parser.parse_args()
    corpus = read_corpus(args.corpus)
    store = FeatureStore(args.resources, args.cache)
    fingerprints = feature_fingerprints(store, corpus.protein_reference_id.unique())
    for path in sorted(Path(args.runs).glob("*/seed_*/fold_*/test_predictions.csv")):
        provenance = json.loads((path.parent/"provenance.json").read_text())
        if (provenance["corpus_sha256"] != digest(args.corpus)
                or provenance["feature_files"] != fingerprints
                or provenance["feature_manifest"] != store.manifest
                or provenance["sequence_sha256"] != digest(Path(args.resources)/"sequences.json")):
            raise ValueError(f"Baseline inputs differ from training: {path}")
        predictions = pd.read_csv(path, float_precision="round_trip")
        assays, summary = matched_llr_comparison(predictions, corpus, store)
        assays.to_csv(path.parent/"matched_llr_metrics.csv", index=False)
        save_json(path.parent/"matched_llr_summary.json", summary)
        print(path.parent, summary, flush=True)


if __name__ == "__main__":
    main()
