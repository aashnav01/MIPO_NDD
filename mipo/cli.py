import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="MIPO-NDD reproducible Colab training")
    commands = p.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--corpus", required=True)
    audit.add_argument("--out", required=True)
    sequences = commands.add_parser("resolve-sequences")
    sequences.add_argument("--corpus", required=True)
    sequences.add_argument("--out", required=True)
    sequences.add_argument("--overrides")
    structures = commands.add_parser("fetch-structures")
    structures.add_argument("--resources", required=True)
    imp = commands.add_parser("import-structure")
    imp.add_argument("--resources", required=True)
    imp.add_argument("--gene", "--reference-id", dest="gene", required=True, help="Sequence record key; use protein_reference_id for V4")
    imp.add_argument("--pdb", required=True)
    imp.add_argument("--chain")
    imp.add_argument("--confidence", choices=["none", "plddt"], default="none")
    imp.add_argument("--allow-subsequence", action="store_true", help="Allow an exact unique contiguous target crop, e.g. removing an affinity tag")
    features = commands.add_parser("features")
    for arg in ["corpus", "resources", "cache"]:
        features.add_argument("--"+arg, required=True)
    features.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    features.add_argument("--window", type=int, default=768)
    features.add_argument("--overlap", type=int, default=128)
    features.add_argument("--no-llr", action="store_true")
    splits = commands.add_parser("splits")
    splits.add_argument("--corpus", required=True)
    splits.add_argument("--out", required=True)
    splits.add_argument("--clusters")
    splits.add_argument("--seed", type=int, default=42)
    splits.add_argument("--test-genes", help="CSV with gene column restricting outer tests and validation/calibration to the NDD target set")
    training = commands.add_parser("train")
    for arg in ["corpus", "resources", "cache", "splits", "config", "out"]:
        training.add_argument("--"+arg, required=True)
    training.add_argument("--fold", type=int, default=0)
    training.add_argument("--metadata")
    training.add_argument("--resume", action="store_true")
    training.add_argument("--workers", type=int, default=0, help="DataLoader worker processes; changes speed, not batches")
    evaluation = commands.add_parser("evaluate")
    for arg in ["corpus", "resources", "cache", "checkpoint", "out"]:
        evaluation.add_argument("--"+arg, required=True)
    evaluation.add_argument("--metadata")
    evaluation.add_argument("--workers", type=int, default=0)
    inference = commands.add_parser("predict")
    for arg in ["variants", "resources", "cache", "checkpoint", "out"]:
        inference.add_argument("--"+arg, required=True)
    inference.add_argument("--metadata")
    inference.add_argument("--export-fields", action="store_true")
    base = commands.add_parser("baselines")
    base.add_argument("--corpus", required=True)
    base.add_argument("--out", required=True)
    base.add_argument("--resources")
    base.add_argument("--cache")
    expand = commands.add_parser("import-proteingym")
    for arg in ["corpus", "reference", "assay-dir", "selection", "output"]:
        expand.add_argument("--"+arg, required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--out", default="artifacts/smoke")
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--runs", nargs="+", required=True)
    aggregate.add_argument("--out", required=True)
    args = vars(p.parse_args())
    command = args.pop("command")
    if command == "audit":
        from .corpus import audit
        print(json.dumps(audit(args["corpus"], args["out"]), indent=2))
    elif command == "resolve-sequences":
        from .resources import resolve_sequences
        resolve_sequences(**args)
    elif command == "fetch-structures":
        from .resources import fetch_structures
        fetch_structures(**args)
    elif command == "import-structure":
        from .resources import import_structure
        import_structure(**args)
    elif command == "features":
        from .features import extract
        args["model_name"] = args.pop("model")
        args["llr"] = not args.pop("no_llr")
        extract(**args)
    elif command == "splits":
        from .splits import make_splits
        make_splits(**args)
    elif command == "train":
        from .train import train
        for before, after in [("splits", "split_file"), ("config", "config_file"), ("fold", "fold_index"), ("metadata", "metadata_path")]:
            args[after] = args.pop(before)
        train(**args)
    elif command == "evaluate":
        from .train import evaluate
        args["metadata_path"] = args.pop("metadata")
        print(json.dumps(evaluate(**args), indent=2))
    elif command == "predict":
        from .inference import infer
        args["metadata_path"] = args.pop("metadata")
        infer(**args)
    elif command == "baselines":
        from .baselines import llr_baselines
        llr_baselines(**args)
    elif command == "import-proteingym":
        from .expansion import import_proteingym
        import_proteingym(**args)
    elif command == "smoke":
        from .smoke import run_smoke
        run_smoke(**args)
    elif command == "aggregate":
        import pandas as pd
        from .common import save_json
        from .metrics import bootstrap_genes
        frames = []
        for run in args["runs"]:
            x = pd.read_csv(Path(run)/"test_metrics.csv")
            x["run"] = str(run)
            frames.append(x)
        all_metrics = pd.concat(frames, ignore_index=True)
        out = Path(args["out"])
        out.mkdir(parents=True, exist_ok=True)
        all_metrics.to_csv(out/"all_metrics.csv", index=False)
        save_json(out/"gene_bootstrap.json", bootstrap_genes(all_metrics))


if __name__ == "__main__":
    main()
