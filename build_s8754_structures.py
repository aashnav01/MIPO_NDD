"""Fetch real coordinates for the S8754 teacher proteins and write project-format structures.

Reads the `s8754_sequences.json` and `s8754_structure_manifest.csv` emitted by
build_s8754_corpus.py. RCSB rows name a PDB entry and chain and get the experimental
structure; UniProt-only rows get AlphaFold, labelled as predicted so analyses can
keep the two apart. Output is `structures/<key>.npz` with the arrays `FeatureStore`
reads (coords, confidence, sequence), plus a per-protein report.

Alignment is gap-tolerant (a PDB chain omits disordered residues) and every accepted
structure records the PDB author residue number of each construct position, so the
mapping can be audited against the S8754 `name` field rather than trusted.

Acceptance: identity over aligned pairs >= --min-identity AND the alignment covers
>= --min-coverage of the construct. Anything else ships sequence-only, which
`FeatureStore` handles with a zero confidence mask; that outcome is reported, never
silent.
"""
from __future__ import annotations

import argparse
import http.client
import json
import time
import warnings
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

AA = set("ACDEFGHIKLMNPQRSTVWY")
AGENT = {"User-Agent": "mipo-research/1.0"}


def download(url, dest, tries=3):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1000:
        return "cached"
    for attempt in range(tries):
        try:
            with urlopen(Request(url, headers=AGENT), timeout=90) as r:
                data = r.read()
            if b"ATOM" not in data:
                return "no ATOM records"
            dest.write_bytes(data)
            return "ok"
        except HTTPError as e:
            if e.code == 404:
                return "404"
            time.sleep(2 * (attempt + 1))
        except (URLError, TimeoutError, http.client.HTTPException, ConnectionError, OSError):
            time.sleep(2 * (attempt + 1))
    return "failed after retries"


def alphafold_url(accession):
    try:
        req = Request(f"https://alphafold.ebi.ac.uk/api/prediction/{accession}", headers=AGENT)
        meta = json.loads(urlopen(req, timeout=60).read())
        return meta[0]["pdbUrl"]
    except (HTTPError, URLError, TimeoutError, KeyError, IndexError, ValueError,
            http.client.HTTPException, OSError):
        return None


def one_letter(residue):
    if residue.resname == "MSE":  # selenomethionine is HETATM but is methionine
        return "M"
    letter = seq1(residue.resname)
    return letter if letter in AA else "X"


def chain_models(pdb_path, chain_id):
    """Per model: [(aa, xyz, resseq, icode, bfactor)] for residues that have a CA."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = PDBParser(QUIET=True).get_structure("s", str(pdb_path))
    models = []
    for model in structure:
        chains = [c.id for c in model]
        use = chain_id if chain_id in chains else (chains[0] if not chain_id and chains else None)
        if use is None:
            continue
        rows = []
        for residue in model[use]:
            if residue.id[0] != " " and residue.resname != "MSE":
                continue
            if "CA" not in residue:
                continue
            ca = residue["CA"]
            rows.append((one_letter(residue), np.asarray(ca.coord, np.float32),
                         int(residue.id[1]), residue.id[2].strip(), float(ca.bfactor)))
        models.append(rows)
    return models


def make_aligner():
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.5
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    # Free end gaps: overhanging construct tails and PDB tags cost nothing. Biopython
    # renamed these attributes, so try the current names first.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:  # current names; assignment only, as reading them raises on mixed gap scores
            aligner.end_insertion_score = 0.0
            aligner.end_deletion_score = 0.0
        except (AttributeError, ValueError):
            aligner.target_end_gap_score = 0.0
            aligner.query_end_gap_score = 0.0
    return aligner


ALIGNER = make_aligner()


def align_chain(sequence, chain_rows):
    """Map construct index -> chain index, with identity/coverage statistics."""
    chain_seq = "".join(r[0] for r in chain_rows)
    if not chain_seq:
        return {}, {"identity": 0.0, "aligned": 0, "chain_len": 0, "seq_len": len(sequence),
                    "seq_coverage": 0.0, "mismatches": 0}
    alignment = ALIGNER.align(sequence, chain_seq)[0]
    mapping, matches = {}, 0
    for (ts, te), (qs, qe) in zip(*alignment.aligned):
        for k in range(te - ts):
            i, j = ts + k, qs + k
            mapping[i] = j
            matches += int(sequence[i] == chain_seq[j] and chain_seq[j] != "X")
    n = len(mapping)
    return mapping, {"identity": matches / max(1, n), "aligned": n, "chain_len": len(chain_seq),
                     "seq_len": len(sequence), "seq_coverage": n / max(1, len(sequence)),
                     "mismatches": n - matches}


def build_arrays(sequence, models, mapping, plddt, max_models=4):
    """coords (M, L, 3), confidence (M, L), plus per-position PDB numbering.

    Confidence is 1.0 at every aligned position for experimental structures and
    pLDDT/100 for AlphaFold. Positions the alignment leaves unplaced get zero
    coordinates and zero confidence, which the loader already excludes.
    """
    L = len(sequence)
    first = models[0]
    kept = [m for m in models[:max_models] if len(m) == len(first)]
    coords = np.zeros((len(kept), L, 3), np.float32)
    conf = np.zeros((len(kept), L), np.float32)
    for m, rows in enumerate(kept):
        for i, j in mapping.items():
            coords[m, i] = rows[j][1]
            conf[m, i] = min(1.0, max(0.0, rows[j][4] / 100.0)) if plddt else 1.0
    numbering = [None] * L
    for i, j in mapping.items():
        numbering[i] = f"{first[j][2]}{first[j][3]}"
    return coords, conf, numbering


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequences", required=True, help="s8754_sequences.json")
    p.add_argument("--manifest", required=True, help="s8754_structure_manifest.csv")
    p.add_argument("--out", required=True, help="Resources dir; structures/ is created inside")
    p.add_argument("--pdb-cache", required=True)
    p.add_argument("--min-identity", type=float, default=0.9)
    p.add_argument("--min-coverage", type=float, default=0.5)
    p.add_argument("--no-alphafold", action="store_true")
    a = p.parse_args()

    records = json.loads(Path(a.sequences).read_text())
    manifest = pd.read_csv(a.manifest, keep_default_na=False)
    cache = Path(a.pdb_cache)
    cache.mkdir(parents=True, exist_ok=True)
    folder = Path(a.out) / "structures"
    folder.mkdir(parents=True, exist_ok=True)

    report = []
    for gene, record in records.items():
        sequence, key = record["sequence"], record["key"]
        entry = {"gene": gene, "key": key, "length": len(sequence), "source": None,
                 "pdb": None, "chain": None, "status": None}
        candidates = manifest[manifest.gene == gene].sort_values("rows", ascending=False)
        best = None
        for c in candidates.itertuples():
            if c.source == "rcsb":
                path = cache / f"{c.entry}.pdb"
                state = download(f"https://files.rcsb.org/download/{c.entry}.pdb", path)
                label, plddt = "rcsb", False
            else:
                if a.no_alphafold:
                    continue
                path = cache / f"AF-{c.entry}.pdb"
                url = alphafold_url(c.entry)
                state = download(url, path) if url else "no AlphaFold record"
                label, plddt = "alphafold", True
            if state not in ("ok", "cached"):
                entry["status"] = f"download: {state}"
                continue
            models = chain_models(path, c.chain)
            if not models:
                entry["status"] = f"chain {c.chain!r} not found in {c.entry}"
                continue
            mapping, stats = align_chain(sequence, models[0])
            score = (stats["identity"] >= a.min_identity and stats["seq_coverage"] >= a.min_coverage)
            if score and (best is None or stats["seq_coverage"] > best[3]["seq_coverage"]):
                best = (label, c, (models, mapping, plddt), stats)
            elif not score:
                entry["status"] = (f"alignment rejected: identity {stats['identity']:.3f}, "
                                   f"coverage {stats['seq_coverage']:.3f}")
        if best is None:
            entry["status"] = entry["status"] or "no candidate"
            entry["status"] = "sequence-only: " + entry["status"]
            report.append(entry)
            continue
        label, c, (models, mapping, plddt), stats = best
        coords, conf, numbering = build_arrays(sequence, models, mapping, plddt)
        np.savez_compressed(folder / f"{key}.npz", coords=coords, confidence=conf,
                            sequence=np.array(sequence))
        (folder / f"{key}.source.json").write_text(json.dumps(
            {"gene": gene, "source": label, "entry": c.entry, "chain": c.chain,
             "conformers": int(len(coords)), "numbering": numbering, **stats}) + "\n")
        entry.update({"source": label, "pdb": c.entry, "chain": c.chain, "status": "ok",
                      "conformers": int(len(coords)), **stats,
                      "covered_positions": int((conf[0] > 0).sum())})
        report.append(entry)
        print(f"{gene:14s} {label:9s} {c.entry:8s} chain {c.chain or '-':2s} "
              f"id {stats['identity']:.3f} cov {stats['seq_coverage']:.3f} "
              f"conformers {len(coords)}", flush=True)

    df = pd.DataFrame(report)
    df.to_csv(folder / "s8754_structure_report.csv", index=False)
    ok = df[df.status == "ok"]
    print(f"\n{len(ok)} of {len(df)} proteins have structures "
          f"({ok.source.value_counts().to_dict()}); "
          f"{int((df.status != 'ok').sum())} sequence-only")
    if (df.status != "ok").any():
        print(df[df.status != "ok"][["gene", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
