"""Domain-level homology dedupe of S8754 teacher proteins against the FireProt benchmark.

Same rule as scripts/verify_fireprot_dedupe.py (local alignment, span >= 50, identity
>= 0.40, flag only when >= 1 benchmark test site lies inside the aligned span), applied
the other way round: it names the S8754 proteins that must not be used as teachers.
The 1.5b `crop` dedupe only catches sequence CONTAINMENT, so distant homologs of a
benchmark protein pass it; this closes that gap.

Writes the flagged S8754 gene list. --strict also drops any pair that crosses the span
and identity thresholds even with no benchmark test site inside.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from Bio.Align import PairwiseAligner


def make_aligner():
    a = PairwiseAligner()
    a.mode = "local"
    a.match_score = 2
    a.mismatch_score = -1
    a.open_gap_score = -2
    a.extend_gap_score = -0.5
    return a


def span_identity(aln, query, target):
    identical, span_q, span_t, qpos = 0, 0, 0, set()
    for (q0, q1), (t0, t1) in zip(*aln.aligned):
        span_q += q1 - q0
        span_t += t1 - t0
        qpos.update(range(int(q0), int(q1)))
        identical += sum(query[i] == target[j] for i, j in zip(range(q0, q1), range(t0, t1)))
    span = min(span_q, span_t)
    return int(span), float(identical / span if span else 0.0), qpos


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-sequences", required=True, help="s8754_sequences.json")
    p.add_argument("--benchmark-sequences", required=True)
    p.add_argument("--benchmark-corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--span", type=int, default=50)
    p.add_argument("--identity", type=float, default=0.40)
    a = p.parse_args()

    teachers = {g: v["sequence"] for g, v in json.loads(Path(a.teacher_sequences).read_text()).items()}
    bench = {g: v["sequence"] for g, v in json.loads(Path(a.benchmark_sequences).read_text()).items()}
    sites = pd.read_csv(a.benchmark_corpus).groupby("gene").ref_pos.apply(set).to_dict()
    aligner = make_aligner()
    rows = []
    for n, (bg, bseq) in enumerate(sorted(bench.items())):
        for tg, tseq in teachers.items():
            aln = aligner.align(bseq, tseq)[0]
            span, ident, qpos = span_identity(aln, bseq, tseq)
            if span >= a.span and ident >= a.identity:
                inside = sorted(int(s) for s in sites.get(bg, set()) if s - 1 in qpos)
                rows.append({"teacher": tg, "benchmark": bg, "span": span, "identity": round(ident, 4),
                             "test_sites_inside": len(inside)})
        print(f"{n + 1}/{len(bench)} {bg}", file=sys.stderr, flush=True)
    d = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "s8754_homology_pairs.csv", index=False)
    flagged = sorted(d[d.test_sites_inside > 0].teacher.unique()) if len(d) else []
    strict = sorted(d.teacher.unique()) if len(d) else []
    # Keyed by sequence hash (mipo.common.seq_key) as well as gene name, so the list
    # survives gene renaming when the corpus is rebuilt after dropping proteins.
    key = lambda gene: hashlib.sha256(teachers[gene].encode()).hexdigest()[:24]
    (out / "s8754_homology_drop.json").write_text(json.dumps(
        {"rule": {"span": a.span, "identity": a.identity, "local alignment": "match 2, mismatch -1, open -2, extend -0.5"},
         "pairs_over_threshold": len(d), "flag_with_test_site": flagged, "strict_any_overlap": strict,
         "flag_keys": [key(g) for g in flagged], "strict_keys": [key(g) for g in strict]}, indent=2) + "\n")
    print(json.dumps({"pairs_over_threshold": len(d), "teachers_flagged_with_test_site": len(flagged),
                      "teachers_flagged_strict": len(strict),
                      "benchmark_proteins_affected": int(d[d.test_sites_inside > 0].benchmark.nunique()) if len(d) else 0}, indent=2))


if __name__ == "__main__":
    main()
