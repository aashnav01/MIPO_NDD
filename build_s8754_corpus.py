"""Turn deduped S8754 teacher variants into corpus rows + a sequences.json fragment.

Input is the output of prepare_s8754_teachers.py (already sign-flipped, deduped
against the FireProt benchmark, condition-averaged and WT-verified). This script
only shapes those variants for `read_corpus` and emits the sequence records the
feature extractor needs.

Decided in S8754_HEAD_DECISION.md: S8754 enters as ORDINARY STABILITY ROWS under
its own assay, not as auxiliary-head targets.

ORIENTATION (the 2026-10-06 bug): the training corpus's stability task is HIGHER = MORE STABLE
(score_orientation higher_domain_abundance_proxy_not_health / higher_source_assay_phenotype_not_health),
which is the OPPOSITE of the FireProt benchmark's ddG > 0 = destabilizing. prepare_s8754_teachers.py
flips S8754 to the benchmark convention, so this script flips it BACK for training, and then proves
the result against the corpus's own stability rows on shared variants (--orientation-check). The aux path can only attach to
corpus rows that already exist, and only 24 of 3,710 teacher variants do.

Three things this script exists to prevent:

1. **A silent no-op.** `train.select_stage_rows` keeps only rows whose
   `supervision_tier` appears in a curriculum stage's `tiers` list. A tier that is
   in no stage trains on nothing, with no error. The tier written here must be
   added to a stage, and `--require-stage-config` checks that against a config file.
2. **A silent exclusion.** Stages with `ndd_filter: unverified` keep only rows
   where `is_direct_ndd != 1`, so these rows carry `is_direct_ndd = 0`.
3. **Leakage through a shared protein.** S8754 proteins that overlap a corpus
   construct are dropped by default, because the corpus gene may be a held-out test
   gene in some fold. `--keep-corpus-overlap` opts out, and then every overlapping
   gene must be checked against the evaluation-gene set by hand.

S8754 is TEACHER ONLY. These genes must never reach a test, validation or
calibration split: generate splits with `mipo splits --test-genes <evaluation
genes csv>` so any gene outside that file stays on the training side by
construction. `make_splits` draws test/validation/calibration only from the
test-gene set, so adding training-only genes leaves every fold's held-out gene
lists untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ASSAY_ID = "s8754_ddg"
CORPUS_OFFSETS = {}   # (teacher sequence, corpus protein id) -> position offset
TIER = "F_STABILITY_FIELD_TEACHER"
SOURCE = "S8754 ddG (sign-flipped to project convention); TEACHER ONLY, never benchmarked"


def seq_key(sequence):
    """Same 24-hex feature-cache key the project uses (mipo.common.seq_key)."""
    return hashlib.sha256(sequence.encode()).hexdigest()[:24]


def corpus_overlap(sequences, corpus_sequences):
    """Map each teacher sequence to a corpus protein_reference_id it overlaps, if any.

    Exact match or either sequence containing the other: the same protein under a
    different construct, tag or truncation.
    """
    hits = {}
    for s in sequences:
        for pid, c in corpus_sequences.items():
            if s == c or s in c or c in s:
                hits[s] = pid
                break
    return hits


def check_orientation(teachers, offsets, corpus_path):
    """Prove the emitted orientation against the corpus's own stability rows, on shared variants.

    `teachers.ddG` is in the BENCHMARK convention (positive = destabilizing); this script emits its
    negation, so the emitted value is -ddG and must correlate POSITIVELY with corpus stability scores
    (higher = more stable). A negative correlation is the 2026-10-06 sign error.

    Must run BEFORE the corpus-overlap drop: the proteins that share variants with the corpus are
    exactly the ones that drop out, so afterwards there is nothing left to check.
    """
    import numpy as np
    from scipy.stats import spearmanr
    corpus = pd.read_csv(corpus_path, low_memory=False,
                         usecols=["protein_reference_id", "wt", "ref_pos", "mut", "score_value", "task"])
    stability = corpus[corpus.task == "stability"]
    if stability.empty:
        raise ValueError(f"{corpus_path} has no stability rows to check orientation against")
    aligned = []
    for (seq, pid), offset in offsets.items():
        mine = teachers[teachers.sequence == seq]
        group = stability[stability.protein_reference_id == pid]
        if mine.empty or group.empty:
            continue
        shifted = mine.assign(ref_pos=mine.ref_pos + offset, emitted=-mine.ddG)
        merged = shifted.merge(group, on=["wt", "ref_pos", "mut"], suffixes=("_teacher", "_corpus"))
        if len(merged):
            aligned.append(merged.assign(corpus_protein=pid))
    if not aligned:
        return {"status": "no shared variants with any corpus stability assay; orientation UNVERIFIED",
                "shared_variants": 0}
    merged = pd.concat(aligned, ignore_index=True)
    rho = float(spearmanr(merged.emitted, merged.score_value).statistic)
    report = {"shared_variants": len(merged), "corpus_proteins": int(merged.corpus_protein.nunique()),
              "spearman_emitted_vs_corpus_stability": round(rho, 4)}
    if not np.isfinite(rho):
        raise ValueError(f"Orientation check undefined on {len(merged)} shared variants")
    if rho < 0:
        raise ValueError(
            f"ORIENTATION ERROR: the values this script would emit correlate {rho:+.3f} with the corpus's own "
            f"stability scores on {len(merged)} shared variants ({merged.corpus_protein.nunique()} proteins). "
            "The corpus stability task is HIGHER = MORE STABLE; these would be trained against every other "
            "stability assay. Flip the sign and re-run.")
    report["status"] = "ok: emitted values agree with the corpus stability convention"
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teachers", required=True, help="s8754_teachers_<level>.csv")
    p.add_argument("--parsed", required=True, help="S8754_parsed.csv, for the entry names")
    p.add_argument("--corpus-sequences", required=True,
                   help="Existing v4/resources/sequences.json, for the overlap check")
    p.add_argument("--evaluation-genes", help="CSV with a gene column: the test-gene set. "
                   "Any emitted gene appearing here is a hard error.")
    p.add_argument("--require-stage-config", help="Training config JSON; asserts the tier "
                   "appears in some stage's tiers list, so the rows cannot be silently dropped")
    p.add_argument("--tier", default=TIER)
    p.add_argument("--homology-drop", help="s8754_homology_drop.json from dedupe_s8754_homology.py. "
                   "Removes teachers that are domain-level relatives of a benchmark protein. "
                   "Containment-only dedupe does not catch near-identical constructs or distant "
                   "homologs, so this is not optional for a run whose result will be benchmarked.")
    p.add_argument("--homology-strict", action="store_true",
                   help="Also drop pairs over the span/identity thresholds with no benchmark test site inside")
    p.add_argument("--allow-no-homology-dedupe", action="store_true",
                   help="Proceed without --homology-drop (benchmark leakage risk; recorded in the report)")
    p.add_argument("--orientation-check", help="Training corpus (curriculum.csv.gz). Measures the emitted "
                   "teacher orientation against the corpus's own stability rows on shared variants and refuses "
                   "to emit anything if they disagree. Skipping it is how the 2026-10-06 sign error shipped.")
    p.add_argument("--allow-no-orientation-check", action="store_true",
                   help="Emit without the orientation check (records the omission in the report)")
    p.add_argument("--min-assay-variants", type=int, default=0,
                   help="Drop teacher proteins with fewer than this many variants. Each protein is its own assay and is "
                        "standardised against itself, so a 1-variant protein gets target exactly 0 and a 2-3 variant one loses "
                        "its level; the sampler also draws proteins uniformly, so tiny ones get far more batches than rows. "
                        "The project's contrast pairing already requires 8 shared variants, hence 8 as the sensible value.")
    p.add_argument("--keep-corpus-overlap", action="store_true",
                   help="Keep S8754 proteins that overlap a corpus construct (then clear "
                        "each against the evaluation-gene set yourself)")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    teachers = pd.read_csv(a.teachers)
    required = {"sequence", "ref_pos", "wt", "mut", "ddG"}
    if required - set(teachers):
        raise ValueError(f"teachers file missing {sorted(required - set(teachers))}")
    parsed = pd.read_csv(a.parsed)
    corpus_sequences = {pid: v["sequence"]
                        for pid, v in json.loads(Path(a.corpus_sequences).read_text()).items()}

    if a.homology_drop:
        spec = json.loads(Path(a.homology_drop).read_text())
        blocked = set(spec["strict_keys" if a.homology_strict else "flag_keys"])
        hit = teachers.sequence.map(seq_key).isin(blocked)
        homology = {"rule": spec["rule"], "level": "strict" if a.homology_strict else "test-site",
                    "proteins_dropped": int(teachers[hit].sequence.nunique()),
                    "variants_dropped": int(hit.sum())}
        teachers = teachers[~hit]
    elif a.allow_no_homology_dedupe:
        homology = {"level": "NONE (explicitly allowed): benchmark leakage risk"}
    else:
        raise ValueError("No --homology-drop given. Containment-only dedupe leaves near-identical and "
                         "homologous benchmark proteins in the teacher set; run dedupe_s8754_homology.py "
                         "and pass its JSON, or pass --allow-no-homology-dedupe to accept the risk.")

    # Offsets of each teacher sequence inside each overlapping corpus construct, for the orientation check.
    global CORPUS_OFFSETS
    CORPUS_OFFSETS = {}
    for seq in teachers.sequence.unique():
        for pid, construct in corpus_sequences.items():
            if seq == construct:
                CORPUS_OFFSETS[(seq, pid)] = 0
            elif seq in construct:
                CORPUS_OFFSETS[(seq, pid)] = construct.index(seq)
            elif construct in seq:
                CORPUS_OFFSETS[(seq, pid)] = -seq.index(construct)

    overlap = corpus_overlap(teachers.sequence.unique(), corpus_sequences)
    if a.orientation_check:
        orientation = check_orientation(teachers, CORPUS_OFFSETS, a.orientation_check)
    elif a.allow_no_orientation_check:
        orientation = {"status": "NOT CHECKED (explicitly allowed)"}
    else:
        raise ValueError("No --orientation-check given. The training corpus's stability task is "
                         "higher = more stable while the benchmark is ddG > 0 = destabilizing; shipping the "
                         "wrong one silently degrades the model (2026-10-06). Pass the training corpus, or "
                         "--allow-no-orientation-check to accept the risk.")
    report = {"orientation": orientation, "homology_dedupe": homology, "teacher_variants_in": len(teachers),
              "teacher_proteins_in": int(teachers.sequence.nunique()),
              "proteins_overlapping_corpus": len(overlap),
              "overlapping_corpus_ids": sorted(set(overlap.values()))}
    if not a.keep_corpus_overlap:
        teachers = teachers[~teachers.sequence.isin(overlap)]
        report["dropped_overlap_variants"] = report["teacher_variants_in"] - len(teachers)
    if a.min_assay_variants:
        size = teachers.groupby("sequence").sequence.transform("size")
        small = size < a.min_assay_variants
        report["min_assay_variants"] = {"threshold": a.min_assay_variants,
                                        "proteins_dropped": int(teachers[small].sequence.nunique()),
                                        "variants_dropped": int(small.sum())}
        teachers = teachers[~small]
    if teachers.empty:
        raise ValueError("No teacher variants left after dropping corpus overlap")

    # One gene id per distinct sequence, named from the source entry. Distinct
    # sequences sharing an entry get a suffix: read_corpus refuses conflicting WT
    # residues at a position within one protein_reference_id.
    entry_of = parsed.groupby("sequence").entry.agg(lambda s: s.mode().iat[0]).to_dict()
    gene_of, used = {}, {}
    for s in sorted(teachers.sequence.unique()):
        base = f"S8754_{entry_of.get(s, seq_key(s)[:8])}"
        n = used.get(base, 0) + 1
        used[base] = n
        gene_of[s] = base if n == 1 else f"{base}_{n}"

    # prepare_s8754_teachers.py emits the BENCHMARK convention (ddG > 0 = destabilizing). The training
    # corpus's stability task is the opposite (higher = more stable), so flip back here. The
    # orientation check below proves it against the corpus rather than trusting this comment.
    rows = teachers.assign(
        gene=teachers.sequence.map(gene_of),
        protein_reference_id=teachers.sequence.map(gene_of),
        score_value=-teachers.ddG,
        score_kind="ddG",
        assay_id=ASSAY_ID,
        assay_type="stability",
        supervision_tier=a.tier,
        is_direct_ndd=0,
        default_sample_weight=1.0,
        source=SOURCE,
    )[["gene", "protein_reference_id", "wt", "ref_pos", "mut", "score_value", "score_kind",
       "assay_id", "assay_type", "supervision_tier", "is_direct_ndd",
       "default_sample_weight", "source", "n_conditions", "ddG_spread"]]

    # read_corpus invariants, asserted here so a bad file never reaches training.
    if rows.duplicated(["protein_reference_id", "ref_pos", "mut"]).any():
        raise ValueError("Duplicate (protein, position, mut) rows")
    conflict = rows.groupby(["protein_reference_id", "ref_pos"]).wt.nunique()
    if (conflict > 1).any():
        raise ValueError("Conflicting WT residues at a position within one protein")
    records = {}
    for s, gene in gene_of.items():
        records[gene] = {"gene": gene, "protein_reference_id": gene, "sequence": s,
                         "key": seq_key(s), "length": len(s), "taxon_id": None,
                         "accession": None, "source": SOURCE,
                         "reported_offset": None, "canonical_offset": None,
                         "canonical_mapping_status": "not_established"}
    bad = [(r.gene, r.wt, r.ref_pos) for r in rows.itertuples()
           if r.ref_pos > len(records[r.gene]["sequence"])
           or records[r.gene]["sequence"][r.ref_pos - 1] != r.wt]
    if bad:
        raise ValueError(f"{len(bad)} rows fail WT verification, e.g. {bad[:5]}")

    if set(records) & set(corpus_sequences):
        raise ValueError("Emitted gene ids collide with existing corpus protein ids")

    if a.evaluation_genes:
        evaluation = set(pd.read_csv(a.evaluation_genes).gene)
        clash = sorted(set(rows.gene) & evaluation)
        if clash:
            raise ValueError(f"Emitted genes are in the evaluation set: {clash}")
        report["evaluation_genes_checked"] = len(evaluation)

    if a.require_stage_config:
        config = json.loads(Path(a.require_stage_config).read_text())
        stages = config.get("stages", [])
        reachable = [s.get("name") for s in stages
                     if "*" in s.get("tiers", []) or a.tier in s.get("tiers", [])]
        if not reachable:
            raise ValueError(
                f"Tier {a.tier} appears in no curriculum stage, so these rows would train on "
                f"NOTHING with no error. Stages are: "
                f"{[(s.get('name'), s.get('tiers')) for s in stages]}. Add the tier to a stage "
                f"(the generic/auxiliary stage is the intended home) and re-run.")
        unverified_only = [s.get("name") for s in stages
                           if s.get("name") in reachable and s.get("ndd_filter") == "verified"]
        report["stages_reached"] = reachable
        if unverified_only:
            report["stages_that_will_drop_these_rows"] = unverified_only
        report["tier_checked_against"] = a.require_stage_config

    # Where each protein's coordinates should come from. RCSB rows name a PDB entry and
    # chain; UniProt rows have no experimental structure and fall back to AlphaFold.
    if "chain" in parsed:
        used = parsed[parsed.sequence.isin(gene_of)].assign(gene=lambda d: d.sequence.map(gene_of))
        manifest = (used.groupby(["gene", "source", "entry", "chain"], dropna=False)
                    .size().rename("rows").reset_index()
                    .sort_values(["gene", "rows"], ascending=[True, False]))
        manifest["chain"] = manifest.chain.fillna("")
    else:
        manifest = None

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        manifest.to_csv(out / "s8754_structure_manifest.csv", index=False)
    rows.to_csv(out / "s8754_corpus_rows.csv", index=False)
    (out / "s8754_sequences.json").write_text(json.dumps(records, indent=2) + "\n")
    report["final"] = {"orientation": "higher = more stable (training corpus stability convention)",
                       "rows": len(rows), "proteins": len(records),
                       "residues_needing_features": int(sum(len(v["sequence"]) for v in records.values())),
                       "tier": a.tier, "assay_id": ASSAY_ID,
                       "mean_score": round(float(rows.score_value.mean()), 4),
                       "destabilizing_share": round(float((rows.score_value < 0).mean()), 4)}
    (out / "s8754_corpus_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
