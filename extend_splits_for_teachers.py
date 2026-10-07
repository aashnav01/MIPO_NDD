"""Add teacher-only genes to an existing expanded split file without touching any held-out role.

Why this exists instead of re-running the split expansion on the merged corpus:
`expand_holdouts` picks validation and calibration groups with a score that depends on how
many training groups support each task (`sum(1 / train_support[t])`). Adding 144 stability
genes changes that support, so a re-expansion can choose DIFFERENT validation and
calibration genes than the pilot did for the same fold. Test genes would stay the same, but
checkpoint selection (validation) and calibration would not, and a 1.5b-versus-v1 comparison
would then mix a data effect with a selection-set effect.

So the pilot's expanded splits are reproduced from the BASE corpus (same source split, same
group counts, same seed), checked against the pilot runs' own recorded folds, and only then
extended here: teacher genes are added as singleton groups to every fold's `train` role.
test / validation / calibration are asserted byte-identical.

The split file also records the corpus sha256, and `train()` refuses a corpus whose digest
differs, so the extended file carries the merged corpus's digest.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", required=True, help="Expanded split file made from the BASE corpus")
    p.add_argument("--corpus", required=True, help="Merged corpus (base + teacher rows)")
    p.add_argument("--teacher-genes", required=True, help="s8754_sequences.json (its keys are the teacher genes)")
    p.add_argument("--mipo-src", default=".", help="Directory containing the mipo package")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    sys.path.insert(0, a.mipo_src)
    from mipo.common import digest, save_json
    from mipo.corpus import read_corpus
    from mipo.splits import verify_fold

    source = json.loads(Path(a.splits).read_text())
    teachers = sorted(json.loads(Path(a.teacher_genes).read_text()))
    d = read_corpus(a.corpus)
    mapping = copy.deepcopy(source["gene_to_group"])

    clash = sorted(set(teachers) & set(mapping))
    if clash:
        raise ValueError(f"Teacher genes already in the split file: {clash[:5]}")
    if set(teachers) & set(mapping.values()):
        raise ValueError("A teacher gene id collides with an existing split group name")
    added_corpus = sorted(set(d.gene) - set(mapping))
    if added_corpus != teachers:
        raise ValueError(f"Corpus genes beyond the split file ({len(added_corpus)}) are not exactly the "
                         f"teacher genes ({len(teachers)}); refusing to guess")
    if set(mapping) - set(d.gene):
        raise ValueError("The merged corpus lost genes that the split file covers")
    protected = set(source.get("holdout_eligible_genes", source.get("evaluation_genes", [])))
    if protected & set(teachers):
        raise ValueError("Teacher genes appear in the evaluation-gene set")

    result = copy.deepcopy(source)
    for gene in teachers:
        mapping[gene] = gene  # singleton group: a teacher is never clustered with a held-out gene
    result["gene_to_group"] = mapping
    for fold, before in zip(result["folds"], source["folds"]):
        fold["train"] = sorted(set(before["train"]) | set(teachers))
        for role in ["test", "validation", "calibration"]:
            assert fold[role] == before[role], f"fold {fold['fold']} {role} changed"
        verify_fold(d, fold, mapping)
    result["corpus_sha256"] = digest(a.corpus)
    result["teacher_extension"] = {"source_sha256": digest(a.splits), "teacher_genes_added": len(teachers),
                                   "roles_touched": ["train"],
                                   "note": "test/validation/calibration identical to the source split"}
    save_json(a.out, result)
    print(json.dumps({"folds": len(result["folds"]), "teacher_genes_added": len(teachers),
                      "held_out_roles_changed": 0, "corpus_sha256": result["corpus_sha256"][:12],
                      "out": str(a.out)}, indent=2))


if __name__ == "__main__":
    main()
