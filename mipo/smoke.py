"""Small, explicitly synthetic pipeline exercise; never a scientific benchmark."""
from pathlib import Path

import numpy as np
import pandas as pd

from .common import AA, save_json, seq_key
from .splits import make_splits


def fixture(out):
    out = Path(out)
    resources, cache = out / "resources", out / "cache"
    resources.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    (resources / "structures").mkdir(exist_ok=True)
    rng = np.random.default_rng(9)
    records, rows = {}, []
    for g in range(5):
        gene = f"SYNTHETIC_{g}"
        seq = "".join(rng.choice(list(AA), 40))
        key = seq_key(seq)
        record = {"sequence": seq, "key": key, "length": len(seq), "source": "SYNTHETIC_TEST_ONLY"}
        records[gene] = record
        folder = cache / key
        folder.mkdir(exist_ok=True)
        np.save(folder / "esm.npy", rng.normal(size=(40, 32)).astype(np.float16))
        np.save(folder / "llr.npy", rng.normal(size=(40, 20)).astype(np.float32))
        save_json(folder / "sequence.json", record)
        coords = np.cumsum(rng.normal(size=(2, 40, 3)), axis=1).astype(np.float32)
        np.savez_compressed(resources / "structures" / f"{key}.npz", coords=coords, confidence=np.ones((2, 40), np.float32), sequence=np.array(seq))
        for p in range(1, 9):
            for shift in [1, 2]:
                wt = seq[p-1]
                mut = AA[(AA.index(wt)+shift) % 20]
                rows.append({"gene": gene, "wt": wt, "ref_pos": p, "mut": mut, "score_value": (p+shift)/12,
                             "score_kind": "collapsed_consensus_y", "assay_id": "synthetic_consensus",
                             "assay_type": "curated_functional_consensus", "supervision_tier": "A_NDD_GOLD"})
    corpus = out / "synthetic.csv"
    pd.DataFrame(rows).to_csv(corpus, index=False)
    save_json(resources / "sequences.json", records)
    save_json(cache / "manifest.json", {"dim": 32, "synthetic": True, "model": "SYNTHETIC_RANDOM_FEATURES"})
    config = {"model": "mipo", "hidden": 16, "mechanisms": 4, "field_layers": 2, "sequence_layers": 1,
              "anchors": 4, "window_radius": 3, "max_nodes": 24, "neighbors": 4, "max_conformers": 2,
              "batch_size": 4, "steps_per_epoch": 3, "epochs": 2, "grad_accum": 2, "lr": .001,
              "seed": 42, "cpu_threads": 1, "dropout": 0., "modality_dropout": 0., "require_llr": True}
    save_json(out / "config.json", config)
    make_splits(corpus, out / "splits.json")
    return corpus, resources, cache, out / "splits.json", out / "config.json"


def run_smoke(out):
    from .train import train
    paths = fixture(out)
    train(*paths, out=Path(out)/"run", allow_synthetic=True)
    print("SYNTHETIC SMOKE PASSED. No real-data accuracy claim.")
