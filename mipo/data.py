from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset, Sampler

from .common import AA, META_FIELDS, TASKS, seq_key


class TargetTransform:
    """Training-only task-scale transform; never estimates held-out assay statistics."""
    def __init__(self, state=None):
        self.state = state or {}

    def fit(self, rows):
        self.state = {}
        for task, d in rows.groupby("task"):
            if task == "consensus":
                center, scale = 0., 1.
            else:
                center = float(d.score_value.median())
                scale = float((d.score_value.quantile(.75)-d.score_value.quantile(.25))/1.349)
                scale = max(scale, .1)
            self.state[task] = {"center": center, "scale": scale, "genes": int(d.gene.nunique())}
        return self

    def transform(self, values, tasks):
        return np.asarray([(float(v)-self.state.get(t, {"center": 0.})["center"])/self.state.get(t, {"scale": 1.})["scale"]
                           for v, t in zip(values, tasks)], np.float32)


class Metadata:
    """Allowlisted semantic metadata. Assay IDs and protein IDs cannot enter tensors."""
    def __init__(self, path=None, vocab=None):
        self.records = {}
        if path:
            d = pd.read_csv(path, keep_default_na=False)
            if d.assay_key.duplicated().any():
                raise ValueError("Duplicate assay metadata keys")
            self.records = d.set_index("assay_key").to_dict("index")
        self.vocab = vocab or {"<unknown>": 0}

    def fit(self, rows):
        for key in sorted(rows.assay_key.unique()):
            for field in META_FIELDS:
                value = str(self.records.get(key, {}).get(field, "unknown"))
                token = field + "=" + value
                if value not in ["", "unknown"] and token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        return self

    def encode(self, key):
        return np.asarray([self.vocab.get(f"{field}={self.records.get(key, {}).get(field, 'unknown')}", 0)
                           for field in META_FIELDS], np.int64)


class FeatureStore:
    def __init__(self, resources, cache, allow_synthetic=False):
        self.resources, self.cache = Path(resources), Path(cache)
        self.records = json.loads((self.resources / "sequences.json").read_text())
        self.manifest = json.loads((self.cache / "manifest.json").read_text())
        if self.manifest.get("synthetic") and not allow_synthetic:
            raise ValueError("Synthetic smoke features cannot be used in a scientific run")

    @lru_cache(maxsize=64)
    def protein(self, gene):
        record = self.records[gene]
        folder = self.cache / record["key"]
        esm = np.load(folder / "esm.npy", mmap_mode="r")
        if esm.shape != (len(record["sequence"]), self.manifest["dim"]):
            raise ValueError(f"{gene}: embedding dimensions mismatch")
        cached = json.loads((folder / "sequence.json").read_text())
        if cached["sequence"] != record["sequence"]:
            raise ValueError(f"{gene}: wrong feature reference sequence")
        llr = np.load(folder / "llr.npy", mmap_mode="r") if (folder / "llr.npy").exists() else None
        if llr is not None and llr.shape != (len(esm), 20):
            raise ValueError("LLR shape mismatch")
        path = self.resources / "structures" / f"{record['key']}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as s:
                if str(s["sequence"]) != record["sequence"]:
                    raise ValueError(f"{gene}: structure sequence mismatch")
                coords, conf = s["coords"].copy(), s["confidence"].copy()
            if coords.ndim != 3 or coords.shape[1:] != (len(esm), 3) or conf.shape != coords.shape[:2]:
                raise ValueError("Structure shape mismatch")
            if not np.isfinite(coords).all() or not np.isfinite(conf).all():
                raise ValueError("Nonfinite structure data")
        else:
            # Missing coordinates have a mask; zero coordinates are never treated as contacts.
            coords = np.zeros((1, len(esm), 3), np.float32)
            conf = np.zeros((1, len(esm)), np.float32)
        return {"esm": esm, "llr": llr, "coords": coords, "confidence": conf,
                "sequence": record["sequence"], "global": np.asarray(esm, np.float32).mean(0)}


class VariantDataset(Dataset):
    def __init__(self, rows, store, transform, metadata, config, load_aux=False):
        self.rows = rows.reset_index(drop=True)
        self.store, self.transform, self.metadata, self.config = store, transform, metadata, config
        self.load_aux = load_aux
        self.targets = transform.transform(self.rows.score_value, self.rows.task)
        self.groups = {key: i for i, key in enumerate(sorted(self.rows.assay_key.unique()))}
        for gene, d in self.rows.groupby("protein_reference_id"):
            seq = store.records[gene]["sequence"]
            for pos, wt in d[["ref_pos", "wt"]].drop_duplicates().itertuples(index=False, name=None):
                if pos > len(seq) or seq[pos-1] != wt:
                    raise ValueError(f"{gene} {wt}{pos}: WT/sequence mismatch")

    def __len__(self):
        return len(self.rows)

    @lru_cache(maxsize=32)
    def graph(self, gene, p):
        prot = self.store.protein(gene)
        L = len(prot["sequence"])
        max_nodes = int(self.config.get("max_nodes", 256))
        half = self.config.get("window_radius", 25)
        local = np.arange(max(0, p-half), min(L, p+half+1))
        if max_nodes and L > max_nodes:
            if max_nodes < 2*half+2:
                raise ValueError("Node budget must exceed local sequence window")
            priority = list(local)
            if self.config.get("structure", True) and prot["confidence"][:, p].max() > 0:
                dist = np.linalg.norm(prot["coords"]-prot["coords"][:, p:p+1], axis=-1).mean(0)
                priority += np.argsort(dist)[:32].tolist()
            priority += np.linspace(0, L-1, max_nodes, dtype=int).tolist()
            priority += list(range(L))
            ids = np.sort(np.asarray(list(dict.fromkeys(priority))[:max_nodes]))
        else:
            ids = np.arange(L)
        site = int(np.flatnonzero(ids == p)[0])
        xyz = prot["coords"][:self.config.get("max_conformers", 4), ids].copy()
        confidence = prot["confidence"][:len(xyz), ids].copy()
        if not self.config.get("structure", True):
            confidence[:] = 0
        k = min(self.config.get("neighbors", 16), len(ids))
        # Union of spatial and sequence neighbors; no global dense L x L allocation.
        spatial_k = max(1, k//2)
        if confidence.max() > 0:
            near = np.concatenate([np.asarray(cKDTree(frame).query(frame, k=spatial_k)[1]).reshape(len(ids), spatial_k)
                                   for frame in xyz], axis=1)
        else:
            near = np.asarray(cKDTree(ids[:, None]).query(ids[:, None], k=spatial_k)[1]).reshape(len(ids), spatial_k)
        neighbors = []
        for i in range(len(ids)):
            seq_near = np.argsort(np.abs(ids-ids[i]))[:k]
            candidates = np.unique(near[i])
            distances = np.linalg.norm(xyz[:, candidates]-xyz[:, i:i+1], axis=-1).mean(0)
            candidates = candidates[np.lexsort((candidates, distances))][:spatial_k]
            neighbors.append(list(dict.fromkeys(candidates.tolist()+seq_near.tolist()))[:k])
        edge = np.asarray(neighbors, np.int64)
        delta = xyz[:, edge]-xyz[:, :, None]
        distances = np.linalg.norm(delta, axis=-1)
        valid = (confidence[:, edge] > .5) & (confidence[:, :, None] > .5)
        valid &= distances < 16.
        valid &= distances > .01
        weight = confidence[:, edge]*confidence[:, :, None]*valid
        unit = delta/np.maximum(distances[..., None], 1e-6)
        edge_features = np.stack([distances.mean(0)/16., distances.std(0)/16.,
                                  np.exp(-distances/8).mean(0),
                                  np.clip(np.abs(ids[edge]-ids[:, None])/32., 0, 1),
                                  weight.mean(0)], -1).astype(np.float32)
        seq_window = np.zeros((2*half+1, prot["esm"].shape[1]), np.float32)
        seq_mask = np.zeros(2*half+1, bool)
        dest = local-p+half
        seq_window[dest] = prot["esm"][local]
        seq_mask[dest] = True
        return {"esm": np.asarray(prot["esm"][ids], np.float32), "global": prot["global"],
                "ids": ids, "site": site, "edge": edge, "edge_features": edge_features,
                "unit": unit.astype(np.float32), "edge_weight": weight.astype(np.float32),
                "confidence": confidence.mean(0), "seq_window": seq_window, "seq_mask": seq_mask}

    def __getitem__(self, index):
        r = self.rows.iloc[index]
        p = int(r.ref_pos)-1
        item = dict(self.graph(r.protein_reference_id, p))
        item["aux_target"] = np.zeros((len(item["ids"]), 2), np.float32)
        item["aux_mask"] = np.zeros((len(item["ids"]), 2), bool)
        if self.load_aux and self.config.get("auxiliary_dir"):
            path = Path(self.config["auxiliary_dir"])/f"{seq_key(r.variant_key)}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as teacher:
                    seq = self.store.records[r.protein_reference_id]["sequence"]
                    if str(teacher["sequence"]) != seq or teacher["target"].shape != (len(seq), 2):
                        raise ValueError("Auxiliary teacher/reference mismatch")
                    item["aux_target"] = teacher["target"][item["ids"]].astype(np.float32)
                    item["aux_mask"] = teacher["mask"][item["ids"]].astype(bool)
                    if not np.isfinite(item["aux_target"][item["aux_mask"]]).all():
                        raise ValueError("Nonfinite supervised auxiliary targets")
        llr = self.store.protein(r.protein_reference_id)["llr"]
        value = float(llr[p, AA.index(r.mut)]) if llr is not None else np.nan
        if self.config.get("require_llr", True) and not np.isfinite(value):
            raise ValueError(f"{r.variant_key}: masked LLR cache missing; run features or disable require_llr for an ablation")
        item.update({"wt": AA.index(r.wt), "mut": AA.index(r.mut),
                     "sample_weight": np.float32(getattr(r, "default_sample_weight", 1.)),
                     "llr": np.asarray([value/10 if np.isfinite(value) else 0., float(np.isfinite(value))], np.float32),
                     "task": TASKS.index(r.task), "meta": self.metadata.encode(r.assay_key),
                     "y": self.targets[index], "group": self.groups[r.assay_key], "row_id": int(r.row_id)})
        return item


def collate(items):
    B, N = len(items), max(len(x["ids"]) for x in items)
    K, C = max(x["edge"].shape[1] for x in items), max(x["unit"].shape[0] for x in items)
    dim = items[0]["esm"].shape[1]
    result = {"esm": np.zeros((B, N, dim), np.float32), "node_mask": np.zeros((B, N), bool),
              "edge": np.zeros((B, N, K), np.int64), "edge_features": np.zeros((B, N, K, 5), np.float32),
              "edge_mask": np.zeros((B, N, K), bool), "unit": np.zeros((B, C, N, K, 3), np.float32),
              "edge_weight": np.zeros((B, C, N, K), np.float32), "confidence": np.zeros((B, N), np.float32),
              "ids": np.full((B, N), -1, np.int64)}
    result["conformer_mask"] = np.zeros((B, C), bool)
    result["aux_target"] = np.zeros((B, N, 2), np.float32)
    result["aux_mask"] = np.zeros((B, N, 2), bool)
    for b, x in enumerate(items):
        n, k, c = len(x["ids"]), x["edge"].shape[1], x["unit"].shape[0]
        result["node_mask"][b, :n] = True
        result["conformer_mask"][b, :c] = True
        result["edge_mask"][b, :n, :k] = True
        for name in ["esm", "confidence", "ids", "aux_target", "aux_mask"]:
            result[name][b, :n] = x[name]
        result["edge"][b, :n, :k] = x["edge"]
        result["edge_features"][b, :n, :k] = x["edge_features"]
        result["unit"][b, :c, :n, :k] = x["unit"]
        result["edge_weight"][b, :c, :n, :k] = x["edge_weight"]
    for name in ["global", "site", "seq_window", "seq_mask", "wt", "mut", "llr", "task", "meta", "y", "group", "row_id", "sample_weight"]:
        result[name] = np.stack([x[name] for x in items])
    return {k: torch.from_numpy(v) for k, v in result.items()}


class BalancedAssayBatches(Sampler):
    """Uniform gene -> uniform assay -> variants. Pairs never cross assay boundaries."""
    def __init__(self, rows, batch_size, steps, seed):
        self.batch_size, self.steps, self.seed, self.epoch = batch_size, steps, seed, 0
        self.groups = {}
        for (gene, assay), d in rows.reset_index(drop=True).groupby(["gene", "assay_key"]):
            self.groups.setdefault(gene, []).append(d.index.to_numpy())
        if not self.groups:
            raise ValueError("No training rows in this curriculum stage")

    def __iter__(self):
        rng = np.random.default_rng(self.seed+self.epoch)
        genes = sorted(self.groups)
        for _ in range(self.steps):
            assays = self.groups[genes[rng.integers(len(genes))]]
            pool = assays[rng.integers(len(assays))]
            yield rng.choice(pool, self.batch_size, replace=len(pool) < self.batch_size).tolist()

    def __len__(self):
        return self.steps
