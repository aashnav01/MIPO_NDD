"""Outer whole-gene/cluster holdouts, distinct validation and calibration groups."""
from pathlib import Path

import numpy as np
import pandas as pd
import copy
import json

from .common import digest, save_json
from .corpus import read_corpus


def make_splits(corpus, out, clusters=None, seed=42, test_genes=None):
    d = read_corpus(corpus)
    genes = sorted(d.gene.unique())
    if clusters:
        c = pd.read_csv(clusters, dtype=str)
        if not {"gene", "cluster"} <= set(c) or c.gene.duplicated().any():
            raise ValueError("Cluster file needs unique gene,cluster rows")
        mapping = c.set_index("gene").cluster.to_dict()
        if set(genes) - set(mapping) or c.cluster.isna().any():
            raise ValueError("Every corpus gene must have a cluster")
    else:
        mapping = dict(zip(genes, genes))
    groups = sorted({mapping[g] for g in genes})
    targets = set(pd.read_csv(test_genes).gene) if test_genes else set(genes)
    if targets-set(genes):
        raise ValueError("Requested test genes are absent from the corpus")
    evaluation_groups = sorted({mapping[g] for g in targets})
    if len(evaluation_groups) < 3:
        raise ValueError("Need at least three NDD evaluation groups for test/validation/calibration")
    if len(groups) < 4:
        raise ValueError("Need at least four groups for train/validation/calibration/test")
    folds = []
    for i, group in enumerate(evaluation_groups):
        rest = [x for x in evaluation_groups if x != group]
        np.random.default_rng(seed + i).shuffle(rest)
        held = {group, rest[0], rest[1]}
        roles = {"test": [group], "validation": rest[:1], "calibration": rest[1:2], "train": [g for g in groups if g not in held]}
        fold = {role: [g for g in genes if mapping[g] in selected] for role, selected in roles.items()}
        fold.update({"fold": i, "held_out_group": group})
        folds.append(fold)
    result = {"corpus_sha256": digest(corpus), "seed": seed, "group_kind": "provided_cluster" if clusters else "gene",
              "gene_to_group": mapping, "evaluation_genes": sorted(targets), "folds": folds}
    save_json(out, result)
    return result


def expand_holdouts(corpus, source, out, validation_groups=5, calibration_groups=3, seed=42):
    """Expand existing holdouts without changing outer tests or splitting clusters.

    Selection uses task presence only, never scores or model performance. Existing
    validation/calibration groups stay in their roles. New groups preferentially
    cover missing tasks, prioritizing scarce tasks on ties and then using seeded
    random tie-breaking, while retaining at
    least one training group for every task currently supported in training.
    """
    if min(validation_groups, calibration_groups) < 1:
        raise ValueError("Holdout group counts must be positive")
    d = read_corpus(corpus)
    result = copy.deepcopy(json.loads(Path(source).read_text()))
    if result["corpus_sha256"] != digest(corpus):
        raise ValueError("Corpus changed after split creation")
    mapping = result["gene_to_group"]
    if set(mapping) != set(d.gene):
        raise ValueError("Split group mapping does not cover corpus genes")
    eligible = {mapping[g] for g in result.get("evaluation_genes", sorted(mapping))}
    task_sets = d.assign(split_group=d.gene.map(mapping)).groupby("split_group").task.agg(set).to_dict()
    all_groups = set(mapping.values())
    for fold in result["folds"]:
        verify_fold(d, fold, mapping)
        roles = {role: {mapping[g] for g in fold[role]}
                 for role in ["train", "validation", "calibration", "test"]}
        rng = np.random.default_rng(seed + fold["fold"])
        order = sorted(eligible & roles["train"])
        rng.shuffle(order)
        priorities = {g: i for i, g in enumerate(order)}
        desired = {"validation": validation_groups, "calibration": calibration_groups}
        if any(len(roles[r]) > n for r, n in desired.items()):
            raise ValueError("Expansion cannot shrink existing holdouts")
        while any(len(roles[r]) < n for r, n in desired.items()):
            for role, count in desired.items():
                if len(roles[role]) >= count:
                    continue
                covered = set().union(*(task_sets[g] for g in roles[role]))
                train_support = {t: sum(t in task_sets[g] for g in roles["train"])
                                 for t in set().union(*(task_sets[g] for g in roles["train"]))}
                candidates = [g for g in order if g in roles["train"]
                              and len(roles["train"]) > 1
                              and all(train_support[t] > 1 for t in task_sets[g])]
                if not candidates:
                    raise ValueError(f"Fold {fold['fold']}: insufficient eligible groups to expand holdouts while retaining task support")
                chosen = min(candidates, key=lambda g: (
                    -len(task_sets[g] - covered),
                    -sum(1/train_support[t] for t in task_sets[g] - covered), priorities[g]))
                roles["train"].remove(chosen)
                roles[role].add(chosen)
        assert set.union(*roles.values()) == all_groups
        for role, groups in roles.items():
            fold[role] = sorted(g for g, group in mapping.items() if group in groups)
        verify_fold(d, fold, mapping)
    result["holdout_expansion"] = {"source_sha256": digest(source), "seed": seed,
                                  "validation_groups": validation_groups, "calibration_groups": calibration_groups,
                                  "strategy": "preserve_existing_then_task_coverage_rare_tasks_seeded_ties_v1"}
    save_json(out, result)
    return result


def verify_fold(d, fold, gene_to_group=None):
    roles = ["train", "validation", "calibration", "test"]
    sets = [set(fold[r]) for r in roles]
    if any(sets[i] & sets[j] for i in range(4) for j in range(i)):
        raise ValueError("Gene leakage across split roles")
    if set.union(*sets) != set(d.gene):
        raise ValueError("Split does not cover exactly the current corpus genes")
    if any(not s for s in sets):
        raise ValueError("Empty split role")
    if gene_to_group is not None:
        groups = [{gene_to_group[g] for g in genes} for genes in sets]
        if any(groups[i] & groups[j] for i in range(4) for j in range(i)):
            raise ValueError("Cluster leakage across split roles")


def fold_task_support(d, fold):
    """Describe task coverage without inspecting any held-out target values."""
    counts = {role: d[d.gene.isin(fold[role])].groupby("task").gene.nunique().to_dict()
              for role in ["train", "validation", "calibration", "test"]}
    return {task: {role: int(values.get(task, 0)) for role, values in counts.items()}
            for task in sorted(d.task.unique())}
