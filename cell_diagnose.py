# DIAGNOSE the crashed run: read-only. Shows which validation assays had an undefined Spearman and why.
import pandas as pd
from pathlib import Path

NDD = Path("/content/drive/MyDrive/MIPO_NDD/v4_15b_s8754")
run = NDD / "runs/mipo/seed_42/fold_57"
table = pd.read_csv(run / "validation_epochs/epoch_000.csv")
cols = ["gene", "assay_key", "task", "n", "spearman", "prediction_std", "target_std"]
print(table[cols].round(4).to_string(index=False))
bad = table[table.spearman.isna()]
print()
print(f"{len(bad)} of {len(table)} validation assays have an undefined Spearman")
print("constant predictions in all of them:", bool((bad.prediction_std < 1e-6).all()) if len(bad) else "n/a")
print("constant targets in any of them:", bool((bad.target_std < 1e-9).any()) if len(bad) else "n/a")
