# Tier 1.5a FireProt results — verified numbers for the TIER15_DESIGN.md amendment

Run 2026-10-05 on Colab GPU. Bundle `f1217eed67cd` (post-exclusion-2427x80), code archive
`MyDrive/MIPO_NDD/mipo_ndd_v4_colab_upload.zip` sha `2205e410237a` (has `branch_probes`).
All 4 `contrast_pilot_v1` checkpoints scored with `--allow-probeless`.

Everything below is computed from the four `fireprot_scores.csv` files by
`fireprot_15a_posthoc.py` (in this folder). The pooled z-Spearman column reproduces
`fireprot_metrics.json` exactly, which is the check that the post-hoc reader agrees with
`mipo/fireprot.pooled_metrics`. Results zip: `fireprot_15a_results-20261005T132211Z-1-001.zip`.

## Headline table

| checkpoint | pooled ρ | z-sign dir acc | **raw-sign dir acc** | ρ w/o 1STN | raw-sign w/o 1STN |
|---|---|---|---|---|---|
| esm_mlp seed_42 fold_12 | 0.4265 | 0.6835 | 0.5563 | 0.3845 | 0.5593 |
| esm_mlp seed_42 fold_57 | 0.4083 | 0.6736 | 0.5504 | 0.3695 | 0.5560 |
| mipo seed_42 fold_12 | 0.4194 | 0.6765 | 0.5512 | 0.3814 | 0.5582 |
| mipo seed_42 fold_57 | 0.4145 | 0.6727 | 0.5554 | 0.3862 | 0.5636 |
| **LLR floor** (identical across all four) | **0.3274** | **0.6347** | **0.5437** | **0.3212** | **0.5494** |

Pooled over 72 of 80 proteins (6 singletons, `FP_1RRO` zero target variance, `FP_1BNL` singleton);
2418 of 2427 rows enter the pooled metric, 2382 the raw-sign metric (rows at ddG == 0 excluded).

## What holds

- **mu beats the zero-shot LLR floor in all four checkpoints**: 0.408–0.427 vs 0.327, i.e. +0.08
  to +0.10. No assay-embedding fitting on benchmark rows; the assay key is unseen. DMS training
  transfers to ddG ranking on held-out proteins.
- Per-protein (n ≥ 5 rows, 53 proteins): mu median ρ 0.344–0.367, IQR roughly [0.10, 0.56],
  beating per-protein LLR in 30–34 of 53 proteins. The spread is wide; the pooled number is not
  a per-protein expectation.
- Dropping the dominant protein costs ~0.04: mu 0.370–0.386 vs LLR 0.321, so the margin narrows
  from ~+0.09 to ~+0.06 but does not vanish. `FP_1STN_1EY0` is 559 of 2427 rows (23%).

## Two findings that change claims

### 1. `direction_accuracy` is not the published predictors' quantity

`mipo/fireprot.py` `pooled_metrics` (L194, L206–207) compares the sign of per-protein **z-scores**
on both sides. That asks whether a variant sits above or below its own protein's mean, not whether
it is stabilizing or destabilizing. The frozen design text says "sign of predicted Δstability vs
sign of −ddG", so the code departs from the frozen contract.

Under the raw-sign definition — the one FoldX / Rosetta / ThermoMPNN papers report — accuracy is
**0.55**, against a 0.544 LLR floor. That is near chance, and published ThermoMPNN-class numbers on
FireProt are roughly 0.75–0.80. The 0.67–0.68 figure must not appear in a table beside those anchors.

The raw-sign reader centres each scorer on its per-protein median before taking the sign, because a
scorer has no absolute zero point across proteins; without centring, the column's arbitrary offset
decides the answer. Rows at exactly ddG == 0 are excluded as having no true direction.

The ranking result (pooled ρ) is unaffected. Only the direction claim is.

### 2. mipo vs esm_mlp is a null

Paired per-protein, matched on fold:

| fold | median ρ difference (mipo − esm_mlp) | mipo better in |
|---|---|---|
| fold_12 | −0.020 | 23 / 53 proteins |
| fold_57 | +0.000 | 26 / 53 proteins |

Pooled mu across all four checkpoints spans 0.4083–0.4265, a spread of 0.018, with esm_mlp holding
both the best (0.4265) and the worst (0.4083). The structure-aware model buys nothing over the
sequence-only MLP on this benchmark. This is consistent with the field-share finding (0.0015–0.015)
and is evidence bearing on the frozen Tier 2 gate, which should now be read against instrument C.

## Caveats to carry into any write-up

- `field_probe` is NaN in all four runs by construction (branch_weight=0 champion, scored with
  `--allow-probeless`). Arm (c) is **unmeasured, not zero**. Post-hoc frozen-trunk probes remain
  the follow-up.
- The LLR sign check (k=3) passes but narrowly: stabilizing mean −7.83 vs destabilizing −9.79 when
  recomputed locally on CPU fp32 with ESM-2 650M. `2TRX D27I` (LLR −15.05) carries the margin, and
  two of the six gate rows have doubtful ddG values (`2ADA W264F` at −8.64, `1RTB Y123A` at +12.0).
  Any future exclusion or replicate change that shifts that tail flips the gate for reasons
  unrelated to the model.
- Report pooled figures with and without `FP_1STN_1EY0` wherever they are quoted.
- The notebook's result-folder naming was fixed before this run: the previous
  `ckpt.parent.parent.name` tag collided across models (`esm_mlp` and `mipo` both produced
  `seed_42_fold_12` / `seed_42_fold_57`), so the second model would have silently overwritten the
  first. Tags are now the full path under `PILOT`, with a uniqueness assert. The same fix is needed
  in whatever generated the notebook.
