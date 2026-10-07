# Tier 1.5b first run: invalid (teacher sign error), and what it still tells us

Run 2026-10-06, folder `v4_15b_s8754_min8s2`, scored to `fireprot_15b_results`
(kept locally as `fireprot_15b_results_wrongsign/`). Four runs, 16 epochs, folds 12 (ASPA) and
57 (KCNQ2), recipe matched to `contrast_pilot_v1`.

**This run is not a valid test of 1.5b.** The teachers went in with the wrong sign. The numbers
below measure what that did, which is useful but is not the 1.5b question.

## The error

The training corpus's stability task is **higher = more stable** (`score_orientation` is
`higher_domain_abundance_proxy_not_health` or `higher_source_assay_phenotype_not_health` for all
537k stability rows). The FireProt benchmark is the opposite: **ddG > 0 = destabilizing**.

`prepare_s8754_teachers.py` verifies the S8754 sign flip against the **benchmark** and flips S8754
to match it. That is correct for scoring and exactly wrong for training, and
`build_s8754_corpus.py` passed the flipped value straight through as `score_value`. So the teachers
were trained anti-correlated with every other stability assay in the corpus.

Measured on the 343 variants that appear in both S8754 and a corpus stability assay, across 10
corpus proteins:

| | Spearman vs corpus stability score |
|---|---|
| value as shipped | **-0.531** |
| raw S8754 (what should have been emitted) | **+0.531** |

The raw S8754 orientation already matched the corpus. The flip was the whole bug.

## What it cost, measured

Pooled z-Spearman of `mu` on the benchmark, paired cluster bootstrap over proteins (300 resamples):

| run | v1 | this run | change | 95% CI |
|---|---|---|---|---|
| mipo fold 12 | 0.419 | 0.381 | **-0.039** | [-0.123, +0.012] |
| mipo fold 57 | 0.415 | 0.349 | **-0.066** | [-0.115, -0.027] |
| esm_mlp fold 12 | 0.427 | 0.368 | **-0.058** | [-0.089, -0.015] |
| esm_mlp fold 57 | 0.408 | 0.327 | **-0.081** | [-0.117, -0.044] |
| *zero-shot LLR floor* | *0.327* | | | |

Three of four CIs exclude zero, so the harm is real and not sampling noise. `esm_mlp` fold 57 landed
on 0.327, exactly the zero-shot LLR floor: that model was pushed back to contributing nothing
beyond the ESM anchor.

## The one thing this run establishes for free

Before it, the open worry was that the treatment was **too weak to detect**: teachers are 2.8% of
stage-1 batches and 23% of stage-2, and a single (model, fold) comparison carries about +/-0.02-0.03.

Wrong-signed teachers moved the benchmark by **0.04-0.08**, well outside that band. So this exposure
is enough to move FireProt measurably. That raises the value of the corrected run: a null result from
it would be a real null, not an underpowered one. It does **not** promise the correct sign helps by a
similar amount; harm and help need not be symmetric.

Also unchanged by the error: **T3 (mipo minus esm_mlp) is still null.** 1.5b gives +0.011 and +0.022
against v1's -0.006 and +0.006, and every difference-in-differences CI spans zero. The structure leg
did not separate even with structured stability supervision, wrongly signed or not.

T2 is not meaningful here: nothing came near the 0.474 probe ceiling.

## The fix, and the guard

`build_s8754_corpus.py` now emits `-ddG` (the training convention) and, more importantly, **proves
it**: `--orientation-check <training corpus>` finds variants shared between the teachers and the
corpus's own stability assays and refuses to emit anything if the correlation is negative. Without
that flag or an explicit `--allow-no-orientation-check`, the script will not run at all.

The check must run **before** the corpus-overlap drop; the proteins that share variants with the
corpus are exactly the ones that get dropped for leakage, so afterwards there is nothing left to
compare against. My first attempt at the guard ran it after the drop and silently found 0 shared
variants.

Verified locally:

- correct sign: 343 shared variants, 10 proteins, Spearman **+0.531**, passes;
- the sign that shipped: **-0.531**, raises `ORIENTATION ERROR`;
- corrected build: 2,170 rows / 50 proteins, mean score -0.758 (higher = more stable), 69%
  destabilizing.

Bundle `05136208d367` carries the fix, and all four notebooks pass `--orientation-check` (the
candidate-listing first pass is exempt, since it emits nothing used for training).

## To redo it

Upload bundle `05136208d367`, re-run `s8754_15b_min8s2.ipynb` with a **new** `TAG` so nothing
overwrites, dry run first (the orientation line should read `+0.531 ... ok`), then `TRAIN_NOW = True`.
About 2 hours of training plus 30-45 minutes of scoring; the data prep is cached.
