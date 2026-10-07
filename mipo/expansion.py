"""Import explicitly selected ProteinGym assays; do not relabel generic data as NDD."""
import re
from pathlib import Path

import pandas as pd

from .common import TYPE_MAP
from .corpus import read_corpus


def import_proteingym(corpus, reference, assay_dir, selection, output):
    """Selection CSV: DMS_id,gene,supervision_tier,assay_type,protein_group.

    Source files must be the official processed DMS CSVs with mutant,DMS_score.
    Non-single substitutions are explicitly counted and excluded.
    """
    base = pd.read_csv(corpus, low_memory=False)
    ref = pd.read_csv(reference).set_index("DMS_id")
    plan = pd.read_csv(selection, keep_default_na=False)
    required = {"DMS_id", "gene", "supervision_tier", "assay_type"}
    if not required <= set(plan):
        raise ValueError(f"Selection needs {required}")
    if plan.DMS_id.duplicated().any():
        raise ValueError("Duplicate selected assay")
    additions = []
    for r in plan.itertuples():
        if r.supervision_tier not in ["D_NDD_FAMILY_TEACHER", "E_GENERIC_HUMAN_DMS", "B_DIRECT_NDD_FUNCTIONAL", "C_DIRECT_NDD_STABILITY"]:
            raise ValueError("Explicit supported evidence tier required")
        if r.assay_type not in TYPE_MAP or r.DMS_id not in ref.index:
            raise ValueError(f"Unknown assay/type: {r.DMS_id}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", r.DMS_id):
            raise ValueError("Unsafe assay filename")
        source = pd.read_csv(Path(assay_dir) / (r.DMS_id+".csv"))
        good = source.mutant.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY][1-9][0-9]*[ACDEFGHIKLMNPQRSTVWY]")
        print(f"{r.DMS_id}: retaining {good.sum()} single substitutions; excluding {(~good).sum()}")
        source = source.loc[good].copy()
        parts = source.mutant.str.extract(r"([A-Z])(\d+)([A-Z])")
        x = pd.DataFrame({"gene": r.gene, "wt": parts[0], "ref_pos": parts[1].astype(int), "mut": parts[2],
                          "score_value": source.DMS_score, "raw_dms_score": source.DMS_score,
                          "score_kind": "ProteinGym_DMS_score", "assay_id": r.DMS_id,
                          "assay_type": r.assay_type, "supervision_tier": r.supervision_tier,
                          "source": "ProteinGym", "assay_level_available": 1,
                          "protein_group": getattr(r, "protein_group", "unknown")})
        x = x[x.wt != x.mut]
        additions.append(x)
    combined = pd.concat([base]+additions, ignore_index=True)
    output = Path(output)
    if output.resolve() == Path(corpus).resolve():
        raise ValueError("Write expanded data to a new file to preserve the original corpus")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pending.csv")
    combined.to_csv(temporary, index=False)
    read_corpus(temporary)  # Reject duplicate existing variant-assay pairs.
    temporary.replace(output)
