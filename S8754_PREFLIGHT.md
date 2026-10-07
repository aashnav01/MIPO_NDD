# S8754 teacher preflight — two blockers for 1.5b

Run 2026-10-05 against `external_data/ddg/ddG_S8754.csv` from the GitHub repo
(8,754 rows, 6 columns: `name, wt_seq, mut_seq, pH, temp, ddG`) and the shipped
FireProt bundle (`corpus_fireprot.csv` `bad4f692b2e8`, 2,427 rows / 80 proteins).
Script output: `S8754_parsed.csv` (per-row parse with sequence-derived positions).

This answers conditions (a), (b) and (c) from the 1.5b teacher decision.
**Both blockers are fixable. Neither should be worked around.**

## BLOCKER 1 — S8754 overlaps the FireProt benchmark

The frozen role boundary: FireProt is BENCHMARK ONLY, "void if it ever touches
training". S8754 as a teacher breaches it directly.

| overlap | count |
|---|---|
| S8754 rows matching a benchmark variant (same protein sequence, position, wt, mut) | 588 |
| **distinct benchmark variants hit** | **432 of 2,427 (17.8%)** |
| benchmark proteins matched exactly by sequence | 19 of 80 |
| benchmark proteins additionally overlapping by construct crop | 45 |
| **union of affected proteins** | **56 of 80** |
| benchmark rows sitting in affected proteins | 1,788 of 2,427 (74%) |

Worst offenders by protein (S8754 rows / benchmark rows): `FP_1STN_1EY0` 218/559,
`FP_2RN2` 190/79, `FP_1WQ5` 82/52, `FP_1POH` 15/13, `FP_1IFB_1IFC_2IFB` 13/14.
Ratios above 1.0 are real: S8754 carries several condition rows per variant.

Note `FP_1STN_1EY0` is the protein that already contributes 23% of benchmark rows
and dominates the 1.5a pooled figure. It is also the single most contaminated.

Training on S8754 unfiltered would put 432 benchmark variants into the training
set and void the 1.5a numbers retroactively. The same domain-level dedupe
discipline already applied for the corpus (`verify_fireprot_dedupe.py`, which
excluded 1QLP/1B5M/1IET) has to run against S8754 before any teacher mapping.

## BLOCKER 2 — the sign convention is inverted

S8754 uses **negative = destabilizing**; the FireProt corpus uses
**positive = destabilizing**. Measured on the 588 overlapping rows:

- `corr(S8754.ddG, FireProt.ddG) = -0.873`
- after flipping the sign: `corr = +0.873`, median absolute residual
  **0.08 kcal/mol**, p90 1.20, max 6.23
- 511 of 588 agree in sign after flipping; 460 of 588 agree within 0.5 kcal/mol

The small median residual confirms these are the same measurements, so the
inversion is a convention difference, not a different quantity. Mapped as-is, every
teacher target would be sign-flipped and the auxiliary head would be trained to
predict the opposite of what it should. The 77 rows that still disagree in sign
after flipping are genuine curation/condition discordance and are worth a look
before they are used as targets.

## Two further things that would corrupt the mapping

**Positions must come from the sequence diff, not the name.** The `name` field
encodes PDB numbering (`rcsb_1A43_A_C218S_7_25`), which is offset from the supplied
construct sequence: `C218S` in that row is position **74** of `wt_seq`. This affects
**1,613 of 8,754 rows (18%)**. The amino-acid identities in the name always agree
with the sequence (8,754/8,754), so the name is reliable for wt/mut and unreliable
for position. All 8,754 rows are exactly one substitution with equal-length
`wt_seq`/`mut_seq`, so the diff is unambiguous. 4 rows carry PDB insertion codes
(e.g. `L27CN`); 518 rows are `uniprot_`-prefixed rather than `rcsb_`.

**One variant, several rows.** 288 proteins, 6,363 distinct variants, 8,754 rows:
1,582 variants appear under more than one pH/temperature, up to 11 rows each.
Replicate ddG spread is median 0.60 kcal/mol but reaches **15.00**. A condition
policy is needed (average, or select one condition) and a spread cap, exactly as
the benchmark build did with `max_replicate_spread`.

Condition fields also contain implausible values: pH is integer-valued 0-11 with
353 rows outside pH 3-10, and temperature runs 0-111 C with 787 rows outside
4-60 C. Zeros look like missing data coded as 0 rather than measurements.

## Both blockers are now cleared in code

`prepare_s8754_teachers.py` (project root) does the dedupe, the sign flip, the
sequence-diff mapping, the condition averaging and the WT verification, and refuses
rather than guessing: it measures the sign flip against the overlap *before*
dropping it, and raises if the flip is unsupported or if dedupe empties the set.
Outputs are in `s8754_teachers/`. Both exclusion levels were run:

| `--exclude` | drops | **teacher variants** | proteins |
|---|---|---|---|
| `exact` | 20 benchmark-matching proteins | **5,239** | 266 |
| `crop` | 56, incl. construct crops | **3,710** | 220 |

The strict choice costs 1,529 variants and 46 proteins and still leaves 3,710
WT-verified variants across 220 proteins. That is ample for an auxiliary head, so
**`crop` is the recommendation**: the benchmark is protected completely and the
teacher set is not meaningfully weakened. The earlier worry that strict dedupe would
make S8754 too small to be worth using does not survive the numbers.

Pipeline detail, `crop`: 8,754 rows parse with 0 rejections (every row is exactly
one substitution and the name's wt/mut always agree with the sequence diff) ->
3,075 rows dropped as benchmark overlap -> 579 dropped for pH/temperature outside
pH 3-10 or 4-60 C -> 3,869 variants after averaging conditions -> 159 dropped by the
2.0 kcal/mol spread cap -> **3,710 final**, all WT-verified.

A useful confirmation that the flip is right: after flipping, mean ddG is +0.92 and
72% of variants are destabilizing, which matches the benchmark's own profile (mean
+1.26, median +0.90, mostly destabilizing). Before flipping it was mean −1.05.

## Recommended order

1. Dedupe S8754 against the 80 benchmark proteins (domain-level, same rule as
   `verify_fireprot_dedupe.py`), and drop the overlap. Decide explicitly whether the
   45 crop-overlap proteins go too; dropping all 56 removes most of the benchmark's
   protein coverage from the teacher set, which may make S8754 too small to be worth
   it. That trade-off is the real 1.5b decision, and it should be frozen in writing
   before any training.
2. Flip the sign to the project convention (ddG > 0 = destabilizing) and assert the
   flip on the overlapping rows, so the check cannot silently regress.
3. Map `(protein, ref_pos, mut)` from the **sequence diff**; WT-verify every row
   against the construct, as the corpus build does.
4. Set the condition policy and spread cap; exclude or repair the pH/temp outliers.
5. Only then wire the auxiliary head. The existing `physical_proxies` path is
   per-residue and 2-channel (contact degree + RMSF from paired WT/mutant
   structures); a scalar per-variant ddG target needs a new head, which is an
   architecture change and contradicts "no architecture change" in the frozen 1.5b
   text. Freeze that decision before building it.

## Amendment 2026-10-06: containment dedupe was not enough

The `crop` yield quoted above (3,710 variants / 220 proteins, recommended as ample) is
**superseded**. `crop` only catches sequence containment. The domain-level rule the
frozen design actually calls for (local alignment, span >= 50, identity >= 0.40, a
benchmark test site inside; `dedupe_s8754_homology.py`) flags 61 further teacher proteins,
including 33 pairs that are >= 90% identical to a benchmark protein but differ by a point
change or tag, so they are not substrings. Together they cover 42 of the 80 benchmark
proteins.

| stage | variants | proteins |
|---|---|---|
| `crop` (as recommended above) | 3,710 | 220 |
| minus 15 proteins overlapping the training corpus | 3,330 | 205 |
| minus 61 domain-level homologs of the benchmark | **2,418** | **144** |

The 432-variant leakage figure above was therefore a **floor**: it counted only
exact-position matches inside identical constructs. Training on the `crop` set would still
have leaked benchmark homologs. `build_s8754_corpus.py` now refuses to run without the
homology result. See `STRUCTURE_PROBE_RESULTS.md` for how much the omission inflated a
transfer estimate (best arm 0.447 -> 0.361).
