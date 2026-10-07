"""Sequential resumable fold/seed/ablation runner for a single Colab GPU."""
import argparse
import json
import os
from pathlib import Path

import torch

from mipo.common import digest, save_json
from mipo.corpus import read_corpus
from mipo.data import FeatureStore
from mipo.train import code_fingerprints, feature_fingerprints, train

ABLATIONS = {
    "esm_mlp": {"model": "esm_mlp", "structure": False, "global_kernel": False, "chemistry": False},
    "esm_chemistry": {"model": "esm_mlp", "structure": False, "global_kernel": False},
    "fixed_heads": {"model": "fixed_heads"},
    "no_structure": {"structure": False},
    "no_global_kernel": {"global_kernel": False},
    "no_llr": {"use_llr": False, "require_llr": False},
    "single_conformer": {"max_conformers": 1},
    "mipo": {},
}


def completed_run_matches(out, config, identity):
    if not (out/"test_summary.json").exists():
        return False
    provenance = json.loads((out/"provenance.json").read_text())
    if provenance["config"] != config or any(provenance.get(k) != v for k, v in identity.items()):
        raise ValueError(f"Completed run inputs/config/code changed; use a new output folder: {out}")
    return all((out/name).exists() for name in ["best.pt", "last.pt", "test_predictions.csv", "test_metrics.csv", "training_status.json"])


def main():
    p = argparse.ArgumentParser()
    for key in ["corpus", "resources", "cache", "splits", "config", "out"]:
        p.add_argument("--"+key, required=True)
    p.add_argument("--metadata")
    p.add_argument("--folds", default="all", help="all or comma-separated fold indices")
    p.add_argument("--seeds", default="42,123,2026")
    p.add_argument("--ablations", default="esm_mlp,mipo")
    p.add_argument("--workers", type=int, default=0, help="DataLoader worker processes; changes speed, not batches")
    p.add_argument("--require-cuda", action="store_true", help="Fail instead of silently training on CPU")
    a = p.parse_args()
    if a.require_cuda and not torch.cuda.is_available():
        raise SystemExit(f"No CUDA GPU visible (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}); "
                         "refusing to train on CPU")
    config = json.loads(Path(a.config).read_text())
    splits = json.loads(Path(a.splits).read_text())
    store = FeatureStore(a.resources, a.cache)
    corpus = read_corpus(a.corpus)
    identity = {"corpus_sha256": digest(a.corpus), "split_sha256": digest(a.splits),
                "metadata_sha256": digest(a.metadata) if a.metadata else None,
                "sequence_sha256": digest(Path(a.resources)/"sequences.json"),
                "feature_manifest": store.manifest, "code_sha256": code_fingerprints(),
                "feature_files": feature_fingerprints(store, corpus.protein_reference_id.unique())}
    folds = range(len(splits["folds"])) if a.folds == "all" else [int(x) for x in a.folds.split(",")]
    for name in a.ablations.split(","):
        if name not in ABLATIONS:
            raise ValueError(f"Unknown ablation {name}; choose {list(ABLATIONS)}")
        for seed in map(int, a.seeds.split(",")):
            c = {**config, **ABLATIONS[name], "seed": seed}
            for fold in folds:
                out = Path(a.out)/name/f"seed_{seed}"/f"fold_{fold:02d}"
                if completed_run_matches(out, c, {**identity, "fold": splits["folds"][fold]}):
                    print(f"Completed: {out}", flush=True)
                    continue
                config_path = out/"config.json"
                if config_path.exists() and json.loads(config_path.read_text()) != c:
                    raise ValueError("Experiment config changed; use a new output folder")
                save_json(config_path, c)
                train(a.corpus, a.resources, a.cache, a.splits, config_path, out, fold,
                      a.metadata, resume=(out/"last.pt").exists(), workers=a.workers)


if __name__ == "__main__":
    main()
