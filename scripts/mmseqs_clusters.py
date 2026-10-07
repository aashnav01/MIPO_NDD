"""Convert MMseqs easy-cluster TSV (representative,member) to strict split groups.

First run: mmseqs easy-cluster proteins.fasta cluster_result tmp_mmseqs
             --min-seq-id 0.3 -c 0.8 --cov-mode 0 --cluster-mode 1
Coverage/identity cutoffs are experimental choices, not a guarantee of no homology.
"""
import argparse
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--tsv", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
d = pd.read_csv(a.tsv, sep="\t", names=["cluster", "gene"], dtype=str)
if d.gene.duplicated().any():
    raise ValueError("Each gene must have exactly one cluster")
d[["gene", "cluster"]].to_csv(a.out, index=False)
