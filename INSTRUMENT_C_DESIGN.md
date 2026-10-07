# Instrument C bundle — the Tier 2 confirmation gate

Eval-only. Measures the **chain-preserved scramble margin** on trained checkpoints
and compares it with the frozen bar of **0.05**. Nothing here trains, and nothing
writes into `v4/` or any run folder.

## What the margin is

```
margin = clean macro_gene_spearman  -  scrambled macro_gene_spearman
```

on the checkpoint's **own test split**, taken from its provenance. `macro_gene_spearman`
is the project's selection metric, so the margin is denominated in the quantity the
runs were selected by. A model that reads geometry loses accuracy when geometry is
corrupted; one that ignores it does not.

## Three modes, one gate

| mode | what it does | role |
|---|---|---|
| `chain_preserved` | keeps backbone edges (`abs(dseq) <= radius`, default 8) bit-identical, rewires the rest | **the gate** — isolates geometry beyond sequence adjacency |
| `full` | rewires every masked edge | topology control; normally the larger margin |
| `edge_ablated` | zeroes every edge weight (silences message passing) | **ceiling diagnostic** — the largest margin any scramble could produce |

The ceiling is what separates two very different null results:

- ceiling large, chain-preserved margin small → the run **has** a usable structure
  path but the geometry beyond the backbone is not carrying signal. The gate has
  failed on its own terms.
- ceiling ~0 → the run does not use geometry at all, so no scramble could ever have
  moved the metric. The gate is **uninformative, not failed**, and the script says so.

## Seeds

`scramble_batch` derives all its RNG from one seed, so a single seed is one
corruption pattern repeated across same-shape batches, not an i.i.d. draw. The
runner therefore averages over several seeds (default `0,1,2,3,4`) and reports the
spread. A gate verdict from one seed is not meaningful.

## Negative control

Any checkpoint with `model: esm_mlp` or `structure: False` is detected by
`structure_blind()`, labelled **NEGATIVE CONTROL**, scored anyway, and excluded from
the gate tally. Its margin must come back ~0. A nonzero margin there means the
instrument is reaching the prediction through something other than geometry, and the
script prints a warning. In the synthetic check such a run also reports
`rewire_attempted = 0`, because a structure-blind graph has only sequence-adjacent
edges and so contains nothing for chain-preserved mode to rewire.

## Provenance

`check_provenance()` applies the same contract as `mipo.train.evaluate`: corpus,
metadata, feature manifest, sequence reference and every feature/structure file digest
must match what the checkpoint was trained on, or the run refuses. A margin can never
be computed against shifted inputs.

## Two implementation notes worth keeping

1. **Scrambling happens on the CPU batch, before the move to device.**
   `scramble_batch` builds its generator on the batch's device and then calls
   `torch.randperm(K, generator=gen)` without a `device=` argument, which raises when
   the generator is a CUDA one. Scrambling first keeps the instrument byte-identical
   between CPU and GPU runs and sidesteps that entirely. If `instruments.py` is ever
   fixed to pass `device=`, this ordering stays correct regardless.

2. **Edge-feature corruption is a slot permutation, not noise.** With small `K` the
   permutation can come out as the identity, contributing no edge-feature change at
   all; in the synthetic fixture (`K = 4`) seed 0 does exactly that, which is why the
   multi-seed default matters. On real runs `K = 16`, so an identity draw is
   negligible — but this is the first thing to suspect if a full-scramble margin looks
   implausibly small. The runner warns when `full < chain_preserved`.

## Verification done before shipping

- 13 tests pass: the 7 existing `tests/test_instruments.py`, plus 6 new
  `tests/test_instrument_c.py` covering the margin arithmetic, the negative-control
  classifier, seed handling, and seed-independence of the ablation.
- End-to-end on a real trained checkpoint from `mipo.smoke.run_smoke`: both a `mipo`
  run and a structure-blind `esm_mlp` run, through the full script including the
  provenance checks, the synthetic-feature guard, and all CSV/JSON outputs.
- **Not** yet run against `contrast_pilot_v3b` or any real checkpoint. The numbers it
  produces there are unmeasured.

## Outputs

Per checkpoint, under `--out/<tag>/`:
- `instrument_c.json` — summary, including `gate_passed` and `gate_statement`
- `instrument_c_per_seed.csv` — one row per mode and seed, with rewire counts
- `instrument_c_per_assay.csv` — paired per-assay deltas against the clean run

And across checkpoints: `instrument_c_summary.csv` / `.json`.
