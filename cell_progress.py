# PROGRESS: per-run epochs, validation score, speed and ETA. Read-only.
import json
import re
import subprocess
from pathlib import Path

BASE = Path("/content/drive/MyDrive/MIPO_NDD/v4_15b_s8754_min8s2")
TOTAL = 16
ps = subprocess.run("ps -eo etimes=,args=", shell=True, capture_output=True, text=True).stdout.splitlines()
elapsed = {}
for line in ps:
    parts = line.split(None, 1)
    if len(parts) < 2 or "run_experiments.py" not in parts[1] or parts[1].startswith("bash"):
        continue
    fold, model = re.search(r"--folds (\d+)", parts[1]), re.search(r"--ablations (\w+)", parts[1])
    if fold and model:
        elapsed[(model.group(1), int(fold.group(1)))] = int(parts[0])
print(f"{'run':20s}{'epochs':>9s}{'stage':>32s}{'val':>8s}{'min/epoch':>11s}{'ETA h':>8s}")
for (model, fold), secs in sorted(elapsed.items()):
    h = BASE / f"runs/{model}/seed_42/fold_{fold:02d}/history.json"
    hist = json.loads(h.read_text()) if h.exists() else []
    done = len(hist)
    per = secs / 60 / done if done else float("nan")
    eta = (TOTAL - done) * per / 60 if done else float("nan")
    last = hist[-1] if hist else {}
    print(f"{model + ' fold ' + str(fold):20s}{done:>6d}/{TOTAL}{last.get('stage', '-'):>32s}"
          f"{last.get('validation_spearman', float('nan')):>+8.3f}{per:>11.1f}{eta:>8.1f}")
print()
print("min/epoch includes about 2-4 min of startup, so it overstates the speed early on and settles after a few epochs.")
