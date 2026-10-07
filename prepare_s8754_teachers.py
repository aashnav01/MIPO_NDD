"""Turn ddG_S8754.csv into 1.5b auxiliary teacher rows, or refuse to.

S8754 is TEACHER ONLY and FireProt is BENCHMARK ONLY, so the whole point of this
script is the dedupe: it drops every S8754 protein that overlaps the benchmark
before anything is written. Run it with --exclude both ways and read the yield
before freezing the choice; the trade-off is real and belongs in the design doc,
not in a default.

Four things the raw file will do to you if mapped naively, each asserted here:

1. The sign convention is inverted. S8754 has negative = destabilizing; the project
   has positive = destabilizing. The flip is verified against the benchmark overlap
   BEFORE the overlap is dropped, so the check cannot silently regress.
2. `name` carries PDB numbering, not sequence positions (1,613 of 8,754 rows differ).
   Positions come from the wt/mut sequence diff; the name supplies only wt/mut
   identities, which are then asserted against the diff.
3. One variant appears under several pH/temperature conditions (up to 11 rows).
   Rows are averaged per variant and a spread cap drops the irreproducible ones.
4. pH and temperature contain zeros that look like missing data coded as 0.

Usage:
  python prepare_s8754_teachers.py --s8754 ddG_S8754.csv \
      --benchmark-corpus corpus_fireprot.csv \
      --benchmark-sequences sequences.json \
      --exclude exact|crop --out teachers/
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

NAME = re.compile(r"^(rcsb|uniprot)_(\w+)_(\w*)_([A-Z])(\d+[A-Z]?)([A-Z])_([\d.]+)_([\d.-]+)$")
PH_RANGE = (3.0, 10.0)
TEMP_RANGE = (4.0, 60.0)


def parse(s8754):
    """One row per substitution, with ref_pos taken from the sequence diff."""
    raw = pd.read_csv(s8754)
    required = {"name", "wt_seq", "mut_seq", "pH", "temp", "ddG"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"S8754 missing columns: {sorted(missing)}")
    fields = raw.name.str.extract(NAME)
    fields.columns = ["source", "entry", "chain", "name_wt", "name_pos", "name_mut", "_ph", "_t"]
    raw = pd.concat([raw, fields], axis=1)

    rows, rejected = [], []
    for r in raw.itertuples():
        if len(r.wt_seq) != len(r.mut_seq):
            rejected.append({"name": r.name, "reason": "unequal sequence lengths"})
            continue
        diff = [i for i, (a, b) in enumerate(zip(r.wt_seq, r.mut_seq)) if a != b]
        if len(diff) != 1:
            rejected.append({"name": r.name, "reason": f"{len(diff)} substitutions, not 1"})
            continue
        i = diff[0]
        # The name's position is PDB numbering and is deliberately not trusted; its
        # amino acids are, so a disagreement there means the row is mislabelled.
        if pd.notna(r.name_wt) and (r.name_wt != r.wt_seq[i] or r.name_mut != r.mut_seq[i]):
            rejected.append({"name": r.name, "reason": "name wt/mut disagree with sequence diff"})
            continue
        rows.append({"name": r.name, "source": r.source, "entry": r.entry, "chain": r.chain,
                     "sequence": r.wt_seq, "ref_pos": i + 1,
                     "wt": r.wt_seq[i], "mut": r.mut_seq[i],
                     "ddG_s8754": r.ddG, "pH": r.pH, "temp_C": r.temp,
                     "name_pos": r.name_pos})
    return pd.DataFrame(rows), pd.DataFrame(rejected)


def overlap(parsed, bench, sequences, level):
    """Benchmark overlap per S8754 protein sequence.

    level='exact' matches identical construct sequences only. level='crop' also
    catches constructs where one sequence contains the other, which is how the same
    protein appears under different tags and truncations.
    """
    by_seq = {v["sequence"]: gene for gene, v in sequences.items()}
    exact = parsed.sequence.map(by_seq)
    genes = exact.copy()
    if level == "crop":
        for seq in parsed.sequence[exact.isna()].unique():
            for bseq, gene in by_seq.items():
                if seq in bseq or bseq in seq:
                    genes[parsed.sequence == seq] = gene
                    break
    paired = parsed.assign(benchmark_gene=exact).dropna(subset=["benchmark_gene"]).merge(
        bench[["gene", "ref_pos", "wt", "mut", "score_value"]],
        left_on=["benchmark_gene", "ref_pos", "wt", "mut"],
        right_on=["gene", "ref_pos", "wt", "mut"], how="inner")
    return genes, paired


def verify_sign_flip(paired):
    """The flip is a claim about the data, so it is measured, not assumed."""
    if len(paired) < 30:
        raise ValueError(f"Only {len(paired)} overlapping rows; too few to verify the sign flip. "
                         "Verify the convention by hand before trusting any teacher target.")
    raw = float(np.corrcoef(paired.ddG_s8754, paired.score_value)[0, 1])
    flipped = float(np.corrcoef(-paired.ddG_s8754, paired.score_value)[0, 1])
    if flipped <= abs(raw) * 0.9 or flipped < 0.5:
        raise ValueError(f"Sign flip not supported: corr raw {raw:+.3f}, flipped {flipped:+.3f}. "
                         "The two sources may not measure the same quantity.")
    residual = (-paired.ddG_s8754 - paired.score_value).abs()
    return {"overlapping_rows": len(paired), "corr_raw": round(raw, 4),
            "corr_flipped": round(flipped, 4),
            "median_abs_residual_kcal": round(float(residual.median()), 4),
            "p90_abs_residual_kcal": round(float(residual.quantile(0.9)), 4),
            "rows_agreeing_within_0.5": int((residual <= 0.5).sum()),
            "rows_same_sign": int((np.sign(-paired.ddG_s8754) == np.sign(paired.score_value)).sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--s8754", required=True)
    p.add_argument("--benchmark-corpus", required=True)
    p.add_argument("--benchmark-sequences", required=True)
    p.add_argument("--exclude", choices=["exact", "crop"], default="crop",
                   help="Which benchmark overlap to drop: exact sequence matches, or "
                        "also construct crops (default, stricter)")
    p.add_argument("--max-spread", type=float, default=2.0,
                   help="Drop variants whose condition rows disagree by more than this (kcal/mol)")
    p.add_argument("--keep-odd-conditions", action="store_true",
                   help="Keep rows outside pH 3-10 / 4-60 C instead of dropping them")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    bench = pd.read_csv(a.benchmark_corpus)
    sequences = json.loads(Path(a.benchmark_sequences).read_text())
    parsed, rejected = parse(a.s8754)
    report = {"s8754_rows_parsed": len(parsed), "s8754_rows_rejected": len(rejected),
              "proteins": int(parsed.sequence.nunique()),
              "name_position_differs_from_sequence": int(
                  (parsed.name_pos.str.extract(r"^(\d+)")[0].astype(float) != parsed.ref_pos).sum()),
              "benchmark_rows": len(bench), "benchmark_proteins": int(bench.gene.nunique())}

    genes, paired = overlap(parsed, bench, sequences, a.exclude)
    report["sign_check"] = verify_sign_flip(paired)
    parsed["ddG"] = -parsed.ddG_s8754  # project convention: ddG > 0 = destabilizing

    distinct = paired[["gene", "ref_pos", "wt", "mut"]].drop_duplicates()
    affected = sorted(set(genes.dropna()))
    report["leakage"] = {
        "exclusion_level": a.exclude,
        "row_level_matches": len(paired),
        "distinct_benchmark_variants_hit": len(distinct),
        "benchmark_variant_share": round(len(distinct) / len(bench), 4),
        "benchmark_proteins_affected": len(affected),
        "affected_proteins": affected}

    kept = parsed[genes.isna()].copy()
    report["after_dedupe"] = {"rows": len(kept), "proteins": int(kept.sequence.nunique()),
                              "rows_dropped": len(parsed) - len(kept)}
    if kept.empty:
        raise ValueError("Dedupe removed every row; S8754 cannot be used as a teacher at this level")

    odd = ((kept.pH < PH_RANGE[0]) | (kept.pH > PH_RANGE[1])
           | (kept.temp_C < TEMP_RANGE[0]) | (kept.temp_C > TEMP_RANGE[1]))
    report["odd_condition_rows"] = int(odd.sum())
    if not a.keep_odd_conditions:
        kept = kept[~odd]

    key = ["sequence", "ref_pos", "wt", "mut"]
    grouped = kept.groupby(key)
    teachers = grouped.ddG.mean().rename("ddG").reset_index()
    teachers["n_conditions"] = grouped.size().values
    teachers["ddG_spread"] = grouped.ddG.agg(lambda x: x.max() - x.min()).values
    wide = teachers.ddG_spread > a.max_spread
    report["variants_before_spread_cap"] = len(teachers)
    report["variants_dropped_by_spread_cap"] = int(wide.sum())
    teachers = teachers[~wide]

    # WT verification: the target must sit on the residue it claims.
    bad = [t.wt for t in teachers.itertuples()
           if t.ref_pos > len(t.sequence) or t.sequence[t.ref_pos - 1] != t.wt]
    if bad:
        raise ValueError(f"{len(bad)} teacher rows fail WT verification against their own sequence")
    report["wt_verified"] = len(teachers)
    report["final"] = {"variants": len(teachers), "proteins": int(teachers.sequence.nunique()),
                       "ddG_mean": round(float(teachers.ddG.mean()), 4),
                       "ddG_destabilizing_share": round(float((teachers.ddG > 0).mean()), 4)}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    teachers.to_csv(out / f"s8754_teachers_{a.exclude}.csv", index=False)
    # The per-row parse carries `entry`, which the corpus builder needs for gene ids.
    parsed.to_csv(out / "s8754_parsed.csv", index=False)
    rejected.to_csv(out / f"s8754_rejected_{a.exclude}.csv", index=False)
    (out / f"s8754_report_{a.exclude}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
