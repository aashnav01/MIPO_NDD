# RESEED: replacement-seed run for the mipo fold 57 run that collapsed at epoch 0 (the bafe protocol: seed 42 -> 1042;
# if 1042 collapses too, use 2042). Run it in the SAME session as the main cell, because it reuses BASE_CMD.
# The collapsed run is moved aside, not deleted. It launches ONE background process.
MODEL, FOLD, SEED, BAD_SEED = "mipo", 57, 1042, 42

import shutil
import subprocess
from pathlib import Path

assert "BASE_CMD" in globals(), "Run the main cell first (TRAIN_NOW = False is enough), in this same session."
BASE = NDD / TAG
bad = BASE / "runs" / MODEL / f"seed_{BAD_SEED}" / f"fold_{FOLD:02d}"
crashed = (bad / "validation_epochs/epoch_000.csv").exists() and not (bad / "history.json").exists() \
    and not (bad / "last.pt").exists() and not (bad / "test_summary.json").exists()
if crashed:
    aside = BASE / "set_aside_collapsed" / f"{MODEL}_seed{BAD_SEED}_fold{FOLD:02d}"
    aside.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(bad), str(aside))
    print("moved the collapsed run aside:", aside)
else:
    print("no crashed seed-42 run to move aside (already moved, or it is a different state); leaving", bad)

cmd = list(BASE_CMD)
cmd[cmd.index("--seeds") + 1] = str(SEED)
cmd[cmd.index("--out") + 1] = str(BASE / "runs_reseeded")
cmd += ["--folds", str(FOLD), "--ablations", MODEL]
running = subprocess.run(["pgrep", "-af", "run_experiments"], capture_output=True, text=True).stdout
same = [ln for ln in running.splitlines() if f"--seeds {SEED}" in ln and f"--folds {FOLD}" in ln and f"--ablations {MODEL}" in ln]
assert not same, "that reseed is already running"
log_dir = BASE / "logs_reseeded"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"{MODEL}_seed{SEED}_fold{FOLD:02d}.log"
proc = subprocess.Popen(cmd, stdout=open(log_path, "a"), stderr=subprocess.STDOUT, start_new_session=True)
print(f"launched {MODEL} seed {SEED} fold {FOLD}: pid {proc.pid} -> {log_path}")
