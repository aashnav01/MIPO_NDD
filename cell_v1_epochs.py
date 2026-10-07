# Did the v1 runs stop early, or run out of epochs? Read-only. Decides whether more epochs can help at all.
import json
from pathlib import Path

V1 = Path("/content/drive/MyDrive/MIPO_NDD/contrast_pilot_v1/runs")
for path in sorted(V1.glob("*/seed_*/fold_*/training_status.json")):
    s = json.loads(path.read_text())
    run = str(path.relative_to(V1).parent)
    print(f"{run:26s} epochs run {s['epochs_run']:>2}/{s['max_epochs']} | best epoch {s['best_epoch']:>2} ({s['best_stage']}) "
          f"| best validation {s['best_validation_spearman']:.3f} | stop: {s['stop_reason']}")
