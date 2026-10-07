"""Audit the S8754 position mapping against real PDB coordinates.

The S8754 `name` field carries a residue number (`rcsb_1A43_A_C218S_...` -> 218). Mapping
variants by that number would be wrong for 1,613 of 8,754 rows, because it is PDB author
numbering and not the index into the construct sequence. The pipeline instead takes the
position from the wt/mut sequence diff. This script checks that choice independently:

  for each variant, the PDB residue that the alignment places at the sequence-diff
  position must carry the SAME author number the `name` gave, and be the SAME amino acid.

If both hold across the board, the sequence-diff mapping and the PDB structure agree on
which residue every variant is at, which is the thing a structure-aware model needs.
Exits nonzero on any disagreement so a bad mapping cannot reach training quietly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parsed", required=True, help="s8754_parsed.csv (needs chain, name_pos)")
    p.add_argument("--sequences", required=True, help="s8754_sequences.json")
    p.add_argument("--structures", required=True, help="Resources dir containing structures/")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    parsed = pd.read_csv(a.parsed, keep_default_na=False)
    records = json.loads(Path(a.sequences).read_text())
    by_seq = {v["sequence"]: (g, v["key"]) for g, v in records.items()}
    folder = Path(a.structures) / "structures"

    sources = {}
    for gene, record in records.items():
        path = folder / f"{record['key']}.source.json"
        if path.exists():
            sources[gene] = json.loads(path.read_text())

    rows = []
    for r in parsed[parsed.source == "rcsb"].itertuples():
        gene = by_seq.get(r.sequence, (None, None))[0]
        src = sources.get(gene)
        if src is None or src["source"] != "rcsb":
            continue
        if (r.entry, r.chain) != (src["entry"], src["chain"]):
            continue  # another entry for the same protein; audited via its own structure only
        label = src["numbering"][r.ref_pos - 1]
        rows.append({"gene": gene, "entry": r.entry, "chain": r.chain, "ref_pos": r.ref_pos,
                     "wt": r.wt, "name_pos": str(r.name_pos), "pdb_number": label,
                     "has_coordinate": label is not None,
                     "numbering_agrees": label is not None and label == str(r.name_pos),
                     "seq_index_equals_name": str(r.ref_pos) == str(r.name_pos)})
    d = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "s8754_mapping_audit.csv", index=False)

    n = len(d)
    summary = {
        "rcsb_rows_audited": n, "proteins": int(d.gene.nunique()),
        "site_has_coordinate": int(d.has_coordinate.sum()),
        "pdb_number_equals_name_number": int(d.numbering_agrees.sum()),
        "sequence_index_equals_name_number": int(d.seq_index_equals_name.sum()),
        "structures_with_residue_mismatches": sorted(g for g, s in sources.items() if s.get("mismatches", 0) > 0),
    }
    (out / "s8754_mapping_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nname number == PDB author number: {summary['pdb_number_equals_name_number']}/{n}   "
          f"(sequence index == name number: {summary['sequence_index_equals_name_number']}/{n})")
    bad = d[~d.numbering_agrees]
    if len(bad) or summary["structures_with_residue_mismatches"]:
        print("\nDISAGREEMENTS:\n", bad.head(15).to_string(index=False))
        sys.exit(1)
    print("mapping audit PASSED: sequence-diff positions and PDB author numbering agree for every audited variant")


if __name__ == "__main__":
    main()
