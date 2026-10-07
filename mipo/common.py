from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

AA = "ACDEFGHIKLMNPQRSTVWY"
TASKS = ["consensus", "activity", "stability", "abundance", "binding", "fitness"]
TYPE_MAP = {"curated_functional_consensus": "consensus", "activity": "activity",
            "stability": "stability", "expression": "abundance", "abundance": "abundance",
            "binding": "binding", "organismal_fitness": "fitness", "fitness": "fitness"}
META_FIELDS = ["host", "selection", "system", "treatment", "region"]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seq_key(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()[:24]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)
