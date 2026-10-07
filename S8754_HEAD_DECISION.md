# 1.5b step 2: the auxiliary-head question dissolves — add rows, not a head

## The finding that decides it

The frozen 1.5b text says: *"Map S8754 + FireProt-train ddG onto (protein, position,
mut) rows as aux targets via the `prepare_field_teachers.py` path."*

That is **not implementable with this data.** The auxiliary path attaches targets to
corpus rows that already exist: `VariantDataset.__getitem__` looks up
`auxiliary_dir/{seq_key(row.variant_key)}.npz` per corpus row, and
`physical_proxies` is a per-node `(N, 2)` field over that row's own graph. It cannot
introduce a protein the corpus does not already contain.

Measured against the v4 corpus (598 protein records):

| | count |
|---|---|
| S8754 teacher proteins (`crop` set) | 220 |
| of those, exactly matching a corpus construct | **2** |
| overlapping a corpus construct by crop | 13 |
| **teacher variants attachable to an existing corpus row** | **24 of 3,710** |

So the aux-head route can reach 0.6% of the teacher set. Building a new scalar head
to consume 24 variants is not worth an architecture change, and the architecture
change was the thing the frozen text wanted to avoid.

## Recommendation: add S8754 as new corpus rows

Give S8754 its own `assay_id` with `assay_type: stability`, and let the existing
assay-conditioned head consume it as ordinary supervision.

Why this is the right shape:

- **It needs no architecture change at all** — not a new head, not a new channel,
  not a loss term. `auxiliary_weight` stays at 0 and `physical_proxies` stays
  untouched. This satisfies "supervision density, not architecture" more literally
  than the aux route would have.
- `TargetTransform` already standardizes each assay on its own median and IQR, so
  S8754's kcal/mol scale needs no special handling.
- `read_corpus` already maps `assay_type: stability` to the `stability` task, which
  the model already trains on.
- The rows also become available to the contrast objective wherever an S8754
  protein carries a second assay.

### What it costs

1. **ESM features for 218 new proteins** — 39,341 residues in total, median length
   136. That is larger than the FireProt 1.5a job (23,412 residues) but the same
   order, so roughly 10-25 minutes of GPU extraction. Cheap.
2. **Structures are optional.** `FeatureStore` already ships sequence-only proteins
   with a zero confidence mask, and instrument C found geometry contributes nothing
   to ranking, so sequence-only is a defensible default. The `name` field carries
   RCSB IDs if structures are ever wanted.
3. **The corpus SHA changes, so the protocol hash changes.** A new run folder, and
   splits regenerate. Results will not be directly comparable to the v1/v3 pilots
   unless the split file is pinned deliberately. This is the real cost and it should
   be decided explicitly, not discovered later.

### The guard that must come with it

S8754 is **teacher only, never benchmarked**. Two things enforce that:

- **Keep S8754 genes out of every test and validation split.** The mechanism already
  exists: `mipo splits --test-genes` restricts outer tests, validation and
  calibration to the NDD target set, so any gene outside that CSV stays on the
  training side. Use it; do not rely on chance.
- **18 corpus proteins overlap the S8754 teacher set**, and the list is uncomfortable:
  `TP53` (2 constructs), `BRCA1`, `SERPINE1`, `CRYGD`, `MYB`, `CALM1`, `CKS1B`,
  `FYN`, `SRC`, `PIN1`, `PRKN`, `PRPF40A`, `YAP1`. `TP53`, `SERPINE1`, `BRCA1`,
  `CRYGD` and `MYB` are the same genes the FireProt build already had to exclude for
  corpus overlap. Before adding anything, check each against the NDD test-gene set:
  an S8754 row on a protein that is a held-out test gene elsewhere is leakage of
  exactly the kind 1.5a was protected from. The safe default is to drop those 18
  proteins' S8754 rows as well, which costs little given 220 proteins.

## What to freeze

1. S8754 is added as corpus rows under a new stability assay, not as aux targets.
   The frozen aux-target sentence is superseded, with the 24-of-3,710 measurement as
   the reason.
2. `--exclude crop` for benchmark dedupe (3,710 variants / 220 proteins).
3. S8754 genes are excluded from test/validation via `--test-genes`, and the 18
   corpus-overlapping proteins are dropped unless each is cleared against the test
   set.
4. Whether splits are regenerated or pinned, stated explicitly.
5. `auxiliary_weight` remains 0 and the `physical_proxies` head is not used in 1.5b.
   The per-residue aux path stays available for genuine paired WT/mutant structure
   teachers, which is what it was built for.

## Amendment 2026-10-06: structures, and the final teacher size

- The 24-of-3,710 measurement that justified "add rows, not a head" still stands.
- The teacher set is **2,418 variants / 144 proteins**, not 3,330 / 205, after the
  domain-level homology dedupe (see the amendment in `S8754_PREFLIGHT.md`). Roughly 17
  variants per protein.
- Item 2 above said sequence-only is a defensible default. It is no longer the plan:
  real coordinates were fetched for all 144 proteins (130 RCSB, 14 AlphaFold) and the
  position mapping was audited against them (3,330 of 3,330 rows agree). The notebook now
  ships them (`WITH_STRUCTURES = True`). Only `mipo` reads them; `esm_mlp` does not.
- That makes 1.5b a two-factor change for the `mipo` arm (denser stability supervision AND
  new structured proteins). Read `esm_mlp` as the supervision-density effect alone and
  `mipo - esm_mlp` as the structure effect, which is exactly the contrast 1.5a found null.
- The feature cache on Drive (`v4_650M_masked`) gains 144 entries, about 70 MB. Existing
  entries are untouched.

## Amendment 2026-10-06 (2): the training recipe must be the v1 recipe

The first notebook printed a training command built from `mipo splits` and the archive's
config file. Neither matches `contrast_pilot_v1`:

- `mipo splits` gives one validation gene and one calibration gene per fold. The pilot used
  `prepare_validation_splits.py` (5 validation + 3 calibration groups, seed 42), and its
  `min_validation_genes` is 5. Fold indices 12 and 57 from a plain split are not the pilot's
  ASPA and KCNQ2 folds.
- The config now comes from the v1 runs' own recorded provenance, and the splits are
  reproduced from the base corpus, compared against v1's recorded folds 12 and 57, and
  only then extended with the teacher genes in the `train` role. Held-out roles are
  asserted identical.

Measured locally: re-expanding on the merged corpus would also have left all 134 folds
unchanged, so the extension is a guarantee rather than a repair. The defect that mattered
was the plain split and the missing pilot config, not selection-set drift.

The notebook warns, and prints "NOT comparable with v1", if the archive's base corpus is
not the corpus v1 trained on. Comparability with the 1.5a baseline depends on that match.

## Amendment 2026-10-06 (3): resume safety and the training treatment

- **Prepared inputs must be byte-identical across Colab sessions.** `train()` resume compares
  the corpus sha, splits sha, sequences sha and the digest of every feature and structure
  file. The first notebook wrote the merged corpus with a plain gzip, which embeds a
  timestamp and filename: the sha changed between two runs on identical inputs
  (`61c5d8017340` then `809c6c434f12`), so a Colab recycle during training would have
  orphaned every partial run. The notebook now writes the corpus with a pinned gzip header,
  and on the first preparation records a fingerprint of every input to
  `MIPO_NDD/v4_15b_s8754/prepared_inputs.json`; later sessions must reproduce it or stop.
- **Treatment size.** Teachers are 0.45% of stage-1 rows but 6.5% of stage-1 batches under
  task-balanced sampling (29% if gene-uniform). Stage 1 is 5 of 16 epochs and the teacher
  tier is excluded from stages 2 and 3, with the backbone frozen in stage 3. Decide
  explicitly whether `F_STABILITY_FIELD_TEACHER` should also be added to stage 2.
- **Success bar.** See `STRUCTURE_PROBE_RESULTS.md` section 6: a stability-calibrated ESM LLR
  probe reaches 0.451 (0.474 with contact) on the benchmark, level with the trained v1
  model (0.446), so 1.5b should be judged against about 0.47, not only against v1.

## Amendment 2026-10-06 (4): a design flaw found after launch, and a correction to my earlier numbers

`mipo` fold 57 (seed 42) collapsed at epoch 0: validation prediction spread was 0.0 in all 9
validation assays, across tasks and genes, so the whole network was emitting a constant. That is
the epoch-0 collapse of the bafe grid (graph models only). It is not a shortage of target spread
(4 teacher rows exceed 10 robust SDs; 1 is clipped). Looking for a data cause found a flaw in
how the teachers enter training:

- Each teacher protein is its own assay and is standardised against itself. The median teacher
  assay has **4 variants**; 32 have **1** (standardised target is exactly 0, so the real ddG is
  discarded), and 40 of 144 sit at the 0.1 scale floor. v1's smallest assay has 63 variants.
- The sampler picks proteins uniformly, so the 94 proteins with fewer than 8 variants (10% of
  the rows) receive **65% of the teacher batches**, and 35% of teacher batches are eight draws of
  the same 1-2 rows. These degenerate batches do not exist in v1 and are a plausible, **unproven**,
  cause of the collapse.
- **Correction.** I said the teachers were "6.5% of stage-1 batches, about three passes per
  variant". Most of that was wasted on the tiny assays. Informative exposure was about 2.2% of
  batches, and a filtered set is only about 1.5 draws per variant over stage 1.

Options, built as separate notebooks (new bundle `648210b5e86b`, which adds `--min-assay-variants`):

| notebook | teacher set | stages | teacher share of batches | draws per variant |
|---|---|---|---|---|
| `s8754_15b.ipynb` (as launched) | 2,418 / 144 proteins | 1 | 6.3% (35% of it degenerate) | 3.1 nominal, ~1 informative |
| `s8754_15b_min8.ipynb` | 2,170 / 50 proteins | 1 | 2.8%, all informative | 1.5 |
| `s8754_15b_min8s2.ipynb` | 2,170 / 50 proteins | 1 and 2 | 2.8% in stage 1, 23% in stage 2 | ~9 |

Stage 2 (`family_teachers`) has only 54 stability genes, so adding the tier makes 50 of them
teachers there. That is a much stronger treatment and a larger departure from the v1 recipe.
Filtering alone removes the degenerate batches but leaves the treatment too light to expect a
detectable effect.

## Decision 2026-10-06 (5): variant C chosen

Teacher set of proteins with >= 8 variants (2,170 variants, 50 proteins) added in stage 1 AND
stage 2 (`s8754_15b_min8s2.ipynb`, run folder `v4_15b_s8754_min8s2`). The as-built run
(`v4_15b_s8754`) is abandoned and kept on Drive as evidence of the as-built version only.

Hardware note: the pipeline is CPU-bound with a single-threaded data pipeline per run, so
throughput is set by vCPUs, not the GPU. Colab Pro+ gives 2 vCPUs on standard runtimes and
up to 8 on High-RAM. The notebook now reads the core count, runs one process per 2 vCPUs
(at most one per run), sets OMP/MKL threads to cores divided by processes, and queues the
remaining runs behind the running ones. On 2 vCPUs the four runs go one at a time.

Pre-stated reading of the result: 1.5b is judged by T1 (vs v1), T2 (vs the 0.474 probe
ceiling) and T3 (mipo minus esm_mlp). Because C raises teacher exposure about sixfold over
the as-built run (about 9 draws per variant against about 1 informative), a null here would
mean stability supervision at this scale does not help, whereas a null from the as-built run
could not have been read.

## Amendment 2026-10-06 (6): 20 epochs and a third fold requested; what that does to comparability

Requested: 20 epochs, 3 folds. Built as `s8754_15b_min8s2_e20.ipynb` (run folder
`v4_15b_s8754_min8s2_e20`).

- **16 was v1's schedule** (5 + 3 + 8). It also sets the cosine learning-rate length
  (`updates = epochs * ceil(steps / accumulation)`), so a different total is a different recipe,
  not just a longer one. 20 = v1's 5:3:8 scaled by 1.25, i.e. **6 + 4 + 10**; edit `STAGE_EPOCHS`
  to change it. Early stopping and patience belong to the final stage only.
- **Third fold: PAX6** (binding + stability assays). Folds are now looked up by test gene, never by
  index: the index of PAX6 differs between the old local corpus (76) and the Drive corpus. v1 never
  ran this fold, so the notebook cross-checks held-out roles only for ASPA and KCNQ2.
- **Consequence:** with a different schedule and a fold v1 never ran, **T1 (vs v1) and T3 (mipo minus
  esm_mlp vs v1) are no longer valid against the existing v1 runs.** They need a matched control arm:
  the same schedule, folds, seeds and config, with the S8754 teacher tier removed and the base
  corpus and base splits (last notebook cell, `RUN_CONTROL`). **T2 (vs the 0.474 probe ceiling)
  stays valid without a control.** The notebook prints `NOT comparable with v1: schedule differs`.
- **Cost:** 6 runs x 20 epochs = 120 run-epochs against 64 for the original plan (1.9x), and the
  control doubles it. The control is a separate wave: starting it with the treatment running would
  put 12 processes on 12 vCPUs.
