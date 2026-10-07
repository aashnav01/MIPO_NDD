# Real coordinates for the S8754 teachers, and what they say about structure

Run 2026-10-06. Bundle `s8754_15b_bundle.zip` (`90cc2d3c25cb`), script `probe_structure.py`,
raw outputs in `s8754_corpus/` and `s8754_probe_local/`.

**Status: complete (2026-10-06, Colab).** Coordinates, the mapping audit, the geometry
correlations and the "beyond ESM" arms with real teacher LLRs are all done. Section 6 holds
the ESM results; sections 2-3 were measured earlier without ESM and still stand.

## 1. Coordinates and mapping — done, and clean

| | |
|---|---|
| teacher proteins after all dedupe | 144 |
| experimental structure (RCSB) | 130 |
| AlphaFold (UniProt-only entries) | 14 |
| sequence-only | **0** |
| residues without coordinates in structured proteins | 960 of 26,198 (3.7%) |

Every PDB chain checked (188, measured on the 205-protein set before the homology drop) aligned to
its S8754 construct at **100% identity with every chain residue placed**, so each construct was built from its own PDB chain and the
structure is the wild type. Coverage of the construct is 70-100% (median 100%); the gap is
disordered residues.

**Mapping audit** (`audit_s8754_mapping.py`, exits nonzero on any disagreement): for every
RCSB variant row, the PDB residue that the alignment places at the sequence-derived
position carries the same author number as the S8754 `name` and the same amino acid.
**3,330 of 3,330 rows agree**, against only 2,826 of 3,330 (85%) if the sequence index were
used as the number. So the sequence-diff positions are independently confirmed, and
`name_pos` is PDB author numbering.

The project's own `VariantDataset.graph` built graphs from these structures without error
(600 sampled rows, 0 errors; median 83% of edges carry real geometry, no graph is
geometry-free). That check ran on the 205-protein set before the homology drop and was not
repeated on the final 144; the structures are produced by the same code.

## 2. Does geometry track ddG? — yes, clearly

Feature: C-alpha contact number at the variant's site within 10 A (count of other CA atoms),
first conformer. This is all the model's graph can see. Experimental structures only: 2,226
variants, 130 proteins. Nothing fitted.

| | pooled z-Spearman with ddG | macro within-protein (47 proteins with >= 8 variants) |
|---|---|---|
| R = 10 A (main) | **+0.300** | **+0.230** |
| R = 8 A | +0.262 | +0.187 |
| R = 12 A | +0.289 | +0.227 |
| after removing the leave-protein-out wt->mut expectation | +0.185 | +0.165 |

| burial tercile | variants | mean ddG | median | share destabilizing |
|---|---|---|---|---|
| exposed | 991 | 0.43 | 0.22 | 62% |
| intermediate | 639 | 0.93 | 0.70 | 76% |
| buried | 596 | 1.32 | 1.23 | 79% |

Monotone, stable across radii, and it survives the removal of substitution identity, so
burial is not just proxying for which amino acids were swapped.

## 3. Transfer to the benchmark — modest, and the homology finding matters

Fit on S8754 teachers only; score FireProt (never fitted). Same pooled z-Spearman function
the 1.5a benchmark used. 67 pooled proteins (74 have coordinates). 95% CIs are cluster
bootstraps over proteins, 1,000 resamples.

| arm | pooled z-Spearman | 95% CI |
|---|---|---|
| substitution identity only, ridge | +0.326 | [+0.262, +0.369] |
| + contact, ridge | **+0.386** | [+0.292, +0.437] |
| substitution identity only, HGB | +0.336 | [+0.275, +0.374] |
| + contact, HGB | +0.361 | [+0.280, +0.416] |
| contact number alone, no fitting | +0.313 | [+0.202, +0.374] |
| *reference, same rows:* zero-shot ESM LLR | +0.341 | |
| *reference, same rows:* trained esm_mlp mu (1.5a, fold_12) | +0.446 | |

Paired effect of adding contact to substitution identity: ridge **+0.055, CI [+0.005, +0.092]**
(excludes zero, narrowly); HGB +0.022, CI [-0.038, +0.070] (includes zero).

Readings that hold: a geometry-only score with no training matches zero-shot ESM (0.313 vs
0.341); contact helps a substitution-only model; and the trained model still beats every
probe here by about 0.06, so the 1.5a "transfer is real" result stands.

### The homology correction (read this before quoting any transfer number)

The first run of this probe used the 205-protein teacher set that had passed the 1.5b
`crop` dedupe, and reported **0.447** for the best arm, apparently matching the trained
model. That was inflated. `crop` only catches sequence *containment*; it let through
near-identical constructs and homologs of benchmark proteins. Applying the repo's own
domain-level rule (local alignment, span >= 50, identity >= 0.40, a benchmark test site
inside) flags **61 of the 205 teacher proteins**, covering **42 of 80 benchmark proteins**.
Of the 84 flagged pairs, **33 are >= 90% identical**.

Removing them (912 variants) moved the arms as follows:

| arm | before | after |
|---|---|---|
| HGB + contact | 0.447 | 0.361 |
| HGB substitution only | 0.391 | 0.336 |
| ridge + contact | 0.423 | 0.386 |

The final teacher set is **2,418 variants over 144 proteins**, and the corpus builder now
refuses to run without the homology result.

## 4. What this does and does not show

It shows that real geometry carries stability signal that is independent of which
substitution was made, that it is recoverable from a single crude feature, and that the S8754
structures are correctly mapped.

It does **not** show that geometry helps a model that already has ESM. ESM embeddings encode
burial implicitly, so contact number may be largely redundant with what the sequence model
already knows. The arms above never include ESM as a fitted feature, so they cannot say.

### Decision rule, fixed before the ESM arms were run

The notebook's probe adds a `substitution + LLR` baseline and a `substitution + LLR +
contact` arm. The structure-feature-probe disjunct of the Tier 2 gate is read as positive
**for stability** only if the paired effect "contact beyond ESM LLR" has a ridge 95% CI
excluding zero **and** a positive HGB point estimate. If the CI includes zero, geometry is
redundant with ESM on this task, which matches instrument C, and the frozen instruction is to
drop the structure leg.

### Why instrument C and this probe are not in conflict

Instrument C scrambled geometry on a trained model and read ranking on DMS *function* genes
(ASPA, KCNQ2); the margin was ~0.0005 against a 0.05 bar. This probe asks a different
question on a different task: whether geometry predicts *stability*. Both can be true. If
the ESM arm comes back positive the implication is not "build Tier 2" but "the current field
encoder is failing to extract signal that exists", which is a diagnosis, not a feature
request.

## 5. Caveats

- One geometric feature, CA contact number. It is the best match for what the graph can see,
  not the best possible burial measure (relative SASA would be sharper).
- The 130 experimental teachers are classic, heavily studied proteins; teachers may be
  homologous to *each other*, which the benchmark-facing dedupe does not address.
- FireProt: `FP_1STN_1EY0` is 23% of benchmark rows; the `w/o 1STN` column in the probe
  output reports it separately. 74 of 80 benchmark proteins have coordinates.
- The trained-model reference is a single checkpoint (esm_mlp fold_12, the best of the four).
- Teacher LLRs do not exist locally: computing them on CPU ran at about 12 sites per minute
  (roughly 1.5 hours), so the run was stopped in favour of Colab.

## 6. Beyond ESM — the real result (Colab, real teacher LLRs, 500 bootstrap resamples)

Same 67 pooled benchmark proteins as section 3; fits use the 2,226 experimental-structure
teacher variants only.

| arm | pooled z-Spearman | 95% CI |
|---|---|---|
| substitution identity only, ridge | +0.326 | [+0.263, +0.366] |
| + contact | +0.386 | [+0.290, +0.436] |
| **+ ESM LLR** | **+0.451** | [+0.371, +0.495] |
| **+ ESM LLR + contact** | **+0.474** | [+0.380, +0.521] |
| substitution identity only, HGB | +0.336 | [+0.271, +0.370] |
| + ESM LLR | +0.368 | [+0.298, +0.419] |
| + ESM LLR + contact | +0.422 | [+0.324, +0.468] |
| *zero-shot ESM LLR, no fit* | +0.341 | [+0.251, +0.413] |
| *contact number alone, no fit* | +0.313 | [+0.193, +0.373] |
| *trained esm_mlp mu (1.5a, fold_12)* | +0.446 | [+0.323, +0.511] |

**Paired effect of adding contact to ESM LLR (the gate-relevant numbers):**

| model | effect | 95% CI | positive in |
|---|---|---|---|
| ridge | +0.021 | [-0.000, +0.037] | 97% |
| HGB | **+0.052** | **[+0.016, +0.081]** | 99% |

On the teachers themselves (experimental, nothing fitted), ESM LLR vs ddG is -0.290 pooled
z-Spearman (negative = predictive) and contact vs ddG is +0.300, while contact vs LLR is only
-0.073. **Geometry and ESM are about equally strong predictors of stability and are nearly
independent of each other.**

### Reading, against the rule fixed before this run

The rule: the structure-probe disjunct is positive for stability only if the ridge CI for
"contact beyond ESM LLR" excludes zero **and** the HGB point estimate is positive.

- HGB leg: clearly met (+0.052, CI excludes zero).
- Ridge leg: **not met by the letter of the rule**. The CI lower bound is -0.000, a hair
  below zero, and the effect is small (+0.021).

So by the rule as written the gate is **not met**, by a rounding-level margin on one leg,
while both model families point the same direction and the independent one is clearly
positive. I am not relaxing the rule after seeing the data. The honest label is
**borderline-positive**: geometry probably adds something beyond ESM for stability, the
amount is small (+0.02 to +0.05 on a base of 0.45-0.47), and this single benchmark cannot
settle it. Note also that the earlier 1,000-resample run put the ridge "beyond substitution"
CI at [+0.005, +0.092] and this 500-resample run at [-0.005, +0.091]: those intervals sit on
the edge of zero and move with the resample count.

### What matters more than the gate: a trivial model matches the trained one

A **42-parameter ridge** (one-hot wt, one-hot mut, one ESM LLR feature) fit on 2,226
teacher variants scores **0.451** on the held-out benchmark, equal to the trained esm_mlp
(0.446) on the same rows. With contact it reaches 0.474. Because the probe can only learn
substitution-level and ESM-level effects, this is not leakage: nothing protein-specific is
memorised.

This changes how 1.5a should be read. 1.5a compared the trained model against the raw LLR
floor (0.327-0.341) and called the gap transfer. A stability-calibrated LLR is a much
higher floor, and the trained model does not clear it. "mu beats zero-shot LLR" still holds;
"the network adds something beyond ESM plus substitution priors" is **not demonstrated**.

### Consequences for 1.5b

- **Set the success bar before training.** A 1.5b `mu` of about 0.45 only recapitulates a
  ridge. To claim the network adds anything it has to clear roughly 0.47 on the same 67
  proteins, in addition to beating v1.
- **The treatment is light.** Under the v1 sampler (`task_balanced_sampling` true) the
  teachers supply **6.5% of stage-1 batches** (29% under gene-uniform sampling), about three
  passes per teacher variant over stage 1, and stage 1 is 5 of 16 curriculum epochs per the
  archive config. Stage 3 freezes the backbone and excludes the teacher tier. Expect a modest
  effect, and consider whether the teacher tier should also join stage 2.
- **`mipo - esm_mlp` is the structure test.** It is the first time the structure leg sees
  structured stability supervision. If it stays null while a one-feature probe gains +0.02 to
  +0.05 beyond ESM, the field encoder is failing to extract available signal.

