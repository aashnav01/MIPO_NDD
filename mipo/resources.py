"""Resolve exact experimental sequences; never silently repair coordinate offsets."""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .common import AA, save_json, seq_key
from .corpus import read_corpus

PG_REFERENCE = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv"


def download(url, params=None):
    for attempt in range(4):
        r = requests.get(url, params=params, timeout=90)
        if r.status_code not in [429, 500, 502, 503, 504]:
            r.raise_for_status()
            return r
        time.sleep(2 ** attempt)
    r.raise_for_status()


def matches(sequence, rows):
    return all(1 <= int(p) <= len(sequence) and sequence[int(p)-1] == wt
               for p, wt in rows[["ref_pos", "wt"]].drop_duplicates().itertuples(index=False, name=None))


def fasta_records(text):
    result, name, chunks = [], None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if name:
                result.append((name, "".join(chunks)))
            name, chunks = line[1:].split()[0], []
        else:
            chunks.append(line.strip())
    if name:
        result.append((name, "".join(chunks)))
    return result


def resolve_sequences(corpus, out, overrides=None):
    """Overrides JSON: gene -> {sequence, accession, source}. All WT sites must agree."""
    d = read_corpus(corpus)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    custom = json.loads(Path(overrides).read_text()) if overrides else {}
    ref_file = out / "proteingym_reference.csv"
    if not ref_file.exists():
        ref_file.write_bytes(download(PG_REFERENCE).content)
    ref = pd.read_csv(ref_file)
    resolved, failures = {}, {}
    previous = json.loads((out / "sequences.json").read_text()) if (out / "sequences.json").exists() else {}
    for gene, rows in d.groupby("protein_reference_id", sort=True):
        gene_symbol = rows.gene.iloc[0]
        if gene in custom:
            candidates = [custom[gene]]
        elif gene in previous and matches(previous[gene]["sequence"], rows):
            candidates = [previous[gene]]
        else:
            assay_ids = set(rows.loc[rows.score_kind != "collapsed_consensus_y", "assay_id"])
            pg = ref[ref.DMS_id.isin(assay_ids)]
            candidates = [{"sequence": str(r.target_seq), "accession": str(getattr(r, "UniProt_ID", "")),
                           "source": "ProteinGym experimental target_seq", "reference": str(r.DMS_id)}
                          for r in pg.itertuples()]
            if not candidates:
                query = f"(gene_exact:{gene_symbol}) AND (organism_id:9606) AND (reviewed:true)"
                data = download("https://rest.uniprot.org/uniprotkb/search", {"query": query, "format": "json", "size": 50}).json()
                candidates = [{"sequence": x["sequence"]["value"], "accession": x["primaryAccession"],
                               "source": "UniProt reviewed canonical"} for x in data.get("results", [])]
                if not any(matches(x["sequence"], rows) for x in candidates):
                    for x in data.get("results", []):
                        fa = download(f"https://rest.uniprot.org/uniprotkb/{x['primaryAccession']}.fasta", {"includeIsoform": "true"}).text
                        candidates.extend({"sequence": seq, "accession": name.split("|")[1], "source": "UniProt reviewed isoform"}
                                          for name, seq in fasta_records(fa))
        valid = {x["sequence"]: x for x in candidates if matches(x["sequence"], rows) and set(x["sequence"]) <= set(AA)}
        if len(valid) != 1:
            failures[gene] = {"matching_unique_sequences": len(valid), "candidate_lengths": [len(x["sequence"]) for x in candidates],
                              "action": "Supply the experimental reference in sequence_overrides.json; do not guess offsets."}
            print(f"{gene}: UNRESOLVED ({len(valid)} matching sequences)", flush=True)
            continue
        record = next(iter(valid.values()))
        record["key"] = seq_key(record["sequence"])
        record["length"] = len(record["sequence"])
        resolved[gene] = record
        save_json(out / "sequences.json", resolved)
        print(f"{gene}: {record['length']} aa, {record['source']}", flush=True)
    save_json(out / "resolution_failures.json", failures)
    if failures:
        raise ValueError(f"Unresolved genes: {list(failures)}; see {out / 'resolution_failures.json'}")
    (out / "proteins.fasta").write_text("".join(f">{g}\n{r['sequence']}\n" for g, r in resolved.items()), encoding="utf-8")
    return resolved


def structure_arrays(pdb_path, sequence, chain=None, confidence="plddt", allow_subsequence=False):
    """Exact-chain sequence match only. PDB model axis is the conformer axis."""
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    parsed = PDBParser(QUIET=True).get_structure("protein", str(pdb_path))
    coords, confs = [], []
    for model in parsed:
        candidates = []
        for c in model:
            if chain is not None and c.id != chain:
                continue
            residues = [r for r in c if r.id[0] == " " and "CA" in r]
            seq = "".join(seq1(r.resname) for r in residues)
            if seq == sequence:
                candidates.append(residues)
            elif allow_subsequence and seq.count(sequence) == 1:
                start = seq.index(sequence)
                candidates.append(residues[start:start+len(sequence)])
        if len(candidates) != 1:
            raise ValueError(f"{pdb_path}: requires one exact sequence-matched chain per conformer; found {len(candidates)}")
        residues = candidates[0]
        coords.append(np.stack([r["CA"].coord for r in residues]))
        confs.append([min(1., max(0., r["CA"].bfactor / 100)) if confidence == "plddt" else 1. for r in residues])
    if not coords:
        raise ValueError("No coordinate models")
    return np.asarray(coords, np.float32), np.asarray(confs, np.float32)


def fetch_structures(resources):
    root = Path(resources)
    sequences = json.loads((root / "sequences.json").read_text())
    folder = root / "structures"
    folder.mkdir(exist_ok=True)
    report = {}
    for gene, record in sequences.items():
        dest = folder / f"{record['key']}.npz"
        if dest.exists():
            report[gene] = "existing exact-sequence cache"
            continue
        try:
            # Resolve ProteinGym entry names to accessions through UniProt when necessary.
            acc = record.get("accession", "")
            if "_" in acc:
                acc = download(f"https://rest.uniprot.org/uniprotkb/{acc}.json").json()["primaryAccession"]
            if not acc:
                raise ValueError("No UniProt accession")
            af = download(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}").json()
            if not af:
                raise ValueError("No AlphaFold record")
            pdb = folder / f"{record['key']}.pdb"
            pdb.write_bytes(download(af[0]["pdbUrl"]).content)
            xyz, confidence = structure_arrays(pdb, record["sequence"])
            np.savez_compressed(dest, coords=xyz, confidence=confidence, sequence=np.array(record["sequence"]))
            report[gene] = "AlphaFold exact full-sequence match"
        except (requests.RequestException, ValueError, KeyError, IndexError) as e:
            report[gene] = f"sequence-only fallback: {e}"
        print(gene, report[gene], flush=True)
    save_json(folder / "report.json", report)


def import_structure(resources, gene, pdb, chain=None, confidence="none", allow_subsequence=False):
    root = Path(resources)
    record = json.loads((root / "sequences.json").read_text())[gene]
    xyz, conf = structure_arrays(pdb, record["sequence"], chain, confidence, allow_subsequence)
    folder = root / "structures"
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder / f"{record['key']}.npz", coords=xyz, confidence=conf, sequence=np.array(record["sequence"]))
    save_json(folder / f"{record['key']}.source.json", {"gene": gene, "source": str(pdb), "chain": chain,
              "confidence": confidence, "exact_subsequence_crop_allowed": allow_subsequence, "conformers": len(xyz)})
