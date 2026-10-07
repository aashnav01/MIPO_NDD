# Instrument C results — Tier 2 confirmation gate: NEGATIVE

Run 2026-10-05 on Colab GPU. Bundle `f0e764cef7bc`, archive `2205e410237a`.
Run set: `contrast_pilot_v3/runs_C` (v3C), both finished checkpoints.
Results zip: `instrument_c_results-20261005T173418Z-1-001.zip`.

## Headline

| checkpoint | test gene | rows | clean | **chain margin (GATE)** | chain sd | full margin | ablated ceiling | gate |
|---|---|---|---|---|---|---|---|---|
| mipo seed_42 fold_12 | ASPA | 11,685 | **0.0016** | 0.0001 | 0.0001 | −0.0011 | −0.0002 | fail |
| mipo seed_42 fold_57 | KCNQ2 | 602 | 0.5257 | 0.0005 | 0.0004 | −0.0006 | +0.0001 | fail |

Bar is 0.05. The margins come in **100x to 500x below it**, and the full-scramble
margins are *negative* — corrupting topology very slightly improved the metric,
which is what noise looks like.

## The instrument did act; the metric did not move

This is not a scramble that failed to fire. Per seed, on fold_12:

- 8,449,925 edges rewired in chain-preserved mode, 47,861,760 in full mode,
  success rate 1.00 in both.
- Predictions did shift: mean `|dmu|` 0.0043 (chain), 0.0156 (full), 0.0047
  (edge-ablated). Targets are standardised to ~1 robust SD per assay, so that is
  a 0.4-1.6% shift in prediction — measurable, and far above float noise.
- Yet `macro_gene_spearman` changed by 0.0001.

So geometry reaches the prediction and then contributes nothing to ranking
quality. That is a stronger statement than "the structure path is disconnected":
the path is live, carries a small amount of signal into `mu`, and none of it is
useful for ordering variants.

The edge-ablated ceiling confirms it from the other side: silencing message
passing outright costs **−0.0002 and +0.0001** of macro Spearman. There was never
any accuracy for a scramble to destroy.

## Two limits on how much this can carry

1. **fold_12 is not a valid test bed.** Its clean `macro_gene_spearman` is
   **0.0016** on ASPA — the run has essentially no predictive signal on its own
   test gene. A margin cannot be read on a model that predicts nothing; you cannot
   lose accuracy that was never there. This looks like the epoch-0 collapse pattern
   seen in the bafe grid and should be checked against its `history.json` and
   `test_summary.json` before the run is used for anything.
   **The gate therefore rests on fold_57 alone.**
2. **fold_57 is one gene, 602 rows, 2 assays.** `macro_gene_spearman` there is a
   single-gene number (KCNQ2), and the per-assay deltas are −0.0003 and −0.0002.
   The verdict is clean in direction but thin in support: one healthy checkpoint,
   one gene.

## Reading, against the frozen gate

The frozen text: Tier 2 gets built only if instrument C (chain-preserved margin
>= 0.05) **or** the explicit structure-feature probe comes back positive; if both
are null, the honest ship drops the structure leg entirely, with the E(3) field as
a documented negative ablation.

Instrument C is **null, decisively in magnitude** on the one checkpoint that could
carry it. It now joins:

- field share 0.0015-0.015 across all arms;
- branch supervision harming mechanism (G = −0.027);
- 1.5a: mipo vs esm_mlp a null (median per-protein difference −0.020 / +0.000,
  mipo better in 23/53 and 26/53 proteins).

Four independent measurements, same direction. The remaining escape hatch is the
**explicit structure-feature probe**, the second disjunct of the gate, which has
not been run. If that is also null, the gate's own instruction is to drop the
structure leg.

What survives untouched: 1.5a showed mu beating the zero-shot LLR floor in all
four checkpoints (0.41-0.43 vs 0.327). Transfer to unseen proteins is real. It
does not appear to come from geometry.

## Caveats carried from the instrument's design

- Each margin is averaged over 5 corruption seeds; the per-seed spread is tiny
  (sd 0.0001-0.0004), so the null is not a seed artifact.
- v3C contains no `esm_mlp` run, so this pass had no structure-blind negative
  control. The edge-ablated ceiling served that role instead.
- Edge-feature corruption is a slot permutation rather than noise. At the real
  `K = 16` an identity draw is negligible, and the observed `|dmu|` shifts confirm
  the corruption landed.
- `full < chain_preserved` here, which normally triggers the slot-permutation
  warning. With both quantities at ~0.001 the ordering is noise, not a signal
  about the instrument.
