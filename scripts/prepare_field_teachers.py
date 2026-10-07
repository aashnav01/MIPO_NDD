"""Build OPTIONAL mutation-specific physical proxy targets from paired ensembles.

Inputs are measured/simulated WT and mutant structures, never coordinate noise.
The model does not infer these labels from an unmutated ensemble alone.
Manifest columns: gene,wt,ref_pos,mut,wt_pdb,mut_pdb (optional chain).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from mipo.common import digest, save_json, seq_key
from mipo.resources import structure_arrays


def contact_degree(coords):
    return np.mean([np.asarray(cKDTree(frame).query_ball_point(frame, r=8., return_length=True))-1 for frame in coords], axis=0)/20.


def rmsf(coords):
    centered = coords-coords.mean(1, keepdims=True)
    reference = centered[0]
    aligned = []
    for frame in centered:
        u, _, vt = np.linalg.svd(frame.T@reference)
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(u@vt)
        aligned.append(frame@u@correction@vt)
    return np.sqrt(np.mean(np.sum((aligned-np.mean(aligned, axis=0))**2, axis=-1), axis=0))/5.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--resources", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    records = json.loads((Path(a.resources)/"sequences.json").read_text())
    rows = pd.read_csv(a.manifest, keep_default_na=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    provenance = []
    for r in rows.itertuples():
        ref_id = getattr(r, "protein_reference_id", "")
        if not ref_id:
            candidates = [key for key, record in records.items() if record.get("gene", key) == r.gene]
            if len(candidates) != 1:
                raise ValueError(f"Specify protein_reference_id for {r.gene}: {len(candidates)} references")
            ref_id = candidates[0]
        if records[ref_id].get("gene", r.gene) != r.gene:
            raise ValueError("Teacher gene/reference mismatch")
        seq = records[ref_id]["sequence"]
        pos = int(r.ref_pos)-1
        if not 0 <= pos < len(seq) or seq[pos] != r.wt:
            raise ValueError("WT mapping mismatch in teacher manifest")
        mutant = seq[:pos]+r.mut+seq[pos+1:]
        chain = getattr(r, "chain", None) or None
        wt, _ = structure_arrays(r.wt_pdb, seq, chain, "none")
        alt, _ = structure_arrays(r.mut_pdb, mutant, chain, "none")
        target = np.zeros((len(seq), 2), np.float32)
        mask = np.zeros_like(target, bool)
        target[:, 0] = contact_degree(alt)-contact_degree(wt)
        mask[:, 0] = True
        if min(len(wt), len(alt)) >= 2:
            target[:, 1] = rmsf(alt)-rmsf(wt)
            mask[:, 1] = True
        variant = f"{ref_id}:{r.wt}{r.ref_pos}{r.mut}"
        np.savez_compressed(out/f"{seq_key(variant)}.npz", target=target, mask=mask, sequence=np.array(seq))
        provenance.append({"variant": variant, "wt_sha256": digest(r.wt_pdb), "mut_sha256": digest(r.mut_pdb),
                           "conformers": [len(wt), len(alt)]})
    save_json(out/"provenance.json", {"channels": ["contact_degree_delta_div20", "aligned_rmsf_delta_div5_angstrom"], "sources": provenance})


if __name__ == "__main__":
    main()
