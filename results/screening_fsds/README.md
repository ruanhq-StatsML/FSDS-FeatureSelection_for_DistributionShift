# Screening-Enhanced FSDS (V1: conditional-mean drift)

Code: `python/screening_fsds.py` (core selection logic is about 30 lines: `scan`, `union`, `select`; `fsds` is the
PermuCATE stage). Stage 1 scores every feature separately in the old and new labelled batches, combines the two
batches at an equal total budget K, then Stage 2 runs cross-fitted PermuCATE (RF nuisances) on the retained columns.

Strategies: `sis_*` plain marginal screening; `csis_*` conditional SIS (X and y residualised on the 10 leading PCs of
each batch); `np_union` quantile-bin NIS; `csis_pooled` one scan of the pooled batches; `csis_ref`/`csis_tgt` single
batch; `csis_inter` intersection; `csis_union` top-K/2 per batch then fill by best cross-batch rank;
`drift_aware` = top K/3 each of U=max(G0,G1), D=|G0-G1|, C=|b0-b1|; `csis_np_union` CSIS union + NIS union;
`isis_union` one ISIS round (ridge on selected set, rescan residuals); `oracle` = all drift coordinates + random fill;
`random`.

DGP: p features with a 10-factor correlation (rho = 0.5), 200 dense weak stable signals (SNR 0.15), 24 drift coordinates
in 5 groups (decay, emerge, reverse: linear; nonmono: x^2; inter: products), equal drift energy per group, total drift
`sd`. n = 1000 per batch. This is only a subset of the specified D0-D9 grid (no D2-D3 correlated-substitute /
suppressor / state-dependent / negative-control / variance-only scenarios).

## Stage 1: drift-support recall (mean over replicates; chance = K/p)

p = 10000, sd = 1.0, 10 reps (`screening_fsds_p10000_sd1.0.txt`)

| method | K=100 recall | K=500 recall | K=500 decay | emerge | reverse | nonmono | inter |
|:--|--:|--:|--:|--:|--:|--:|--:|
| sis_union | 0.13 | 0.23 | 0.40 | 0.46 | 0.12 | 0.10 | 0.02 |
| csis_ref | 0.25 | 0.33 | 0.98 | 0.04 | 0.46 | 0.08 | 0.05 |
| csis_tgt | 0.26 | 0.33 | 0.10 | 0.96 | 0.50 | 0.04 | 0.00 |
| csis_pooled | 0.24 | 0.35 | 0.80 | 0.76 | 0.06 | 0.06 | 0.02 |
| csis_inter | 0.01 | 0.08 | 0.10 | 0.04 | 0.24 | 0.00 | 0.00 |
| csis_union | 0.43 | 0.53 | 0.98 | 0.92 | 0.58 | 0.08 | 0.00 |
| drift_aware | 0.45 | 0.57 | 0.96 | 0.96 | 0.72 | 0.08 | 0.02 |
| csis_np_union | 0.39 | 0.54 | 0.98 | 0.92 | 0.60 | 0.08 | 0.00 |
| isis_union | 0.42 | 0.54 | 1.00 | 0.92 | 0.58 | 0.08 | 0.02 |
| random | 0.00 | 0.03 | | | | | |

p = 50000, sd = 1.0, 5 reps: K=200 / K=1000 recall: sis_union 0.11 / 0.18, csis_pooled 0.17 / 0.27,
csis_ref 0.21 / 0.27, csis_union 0.36 / 0.46, drift_aware 0.36 / 0.43, csis_np_union 0.30 / 0.44.

Weak drift (sd = 0.2, `screening_fsds_p10000_sd0.2.txt`): all methods collapse (K=100: csis_union 0.11, drift_aware
0.11, random 0.00; K=500: csis_union 0.22, drift_aware 0.21, random 0.03). Screening has no free lunch when per-coordinate drift is below the marginal
noise level.

Cost (p = 10000 / 50000, both batches): SIS and CSIS 0.3 s / 0.8 s, NIS 2.1 s / 8.4 s.

## Stage 2: FSDS accuracy after screening (p = 10000, K = 500, sd = 1.0, 5 reps, mean (s.e.))

AUC over all p with unretained features ranked last; FSDS seconds exclude screening (screening is 2-3 s).

| method | FSDS AUC | FSDS seconds |
|:--|--:|--:|
| no_screen (all 10000) | 0.577 (0.036) | 24.0 |
| random 500 | 0.483 (0.005) | 6.3 |
| sis_union | 0.594 (0.027) | 6.2 |
| csis_pooled | 0.641 (0.019) | 6.2 |
| csis_union | 0.739 (0.020) | 6.2 |
| drift_aware | 0.754 (0.019) | 6.2 |
| csis_np_union | 0.754 (0.018) | 5.8 |
| oracle screen | 0.980 (0.001) | 6.3 |

## Reading

1. Conditioning on the core factors is the big lever: plain SIS union 0.23 vs CSIS union 0.53 recall (K = 500),
   because factor-induced marginal correlation floods the plain SIS ranking.
2. Two-batch screening matters: single-batch and pooled screening each miss one drift type (decay or emerge) by
   construction, and the intersection is essentially empty. Union at equal budget (or drift-aware U/D/C) recovers
   decay/emerge about 0.95 and reverse 0.6-0.7.
3. Nonmonotone and interaction drift are not recovered by any linear or 5-bin NIS rule here (<= 0.12), so screening
   restricts FSDS to the part of the drift the screens can see. This is the estimand-change caveat of the spec:
   FSDS after screening answers a question about the retained set, not about all p; there is no universal sure
   screening guarantee, and `oracle` is the unattainable ceiling.
4. Net effect: FSDS 0.58 -> 0.74-0.75 AUC at 4x lower FSDS time. Part of this AUC gain is mechanical (dropped features
   are ranked last, and recall of the 24 drift coordinates at K = 500 is 0.55); top-k recall of FSDS itself stays low
   (0.08-0.14 vs 0.19 for oracle), so the within-candidate ranking is still the weak step.
5. drift_aware vs csis_union is within noise in these runs; the extra U/D/C machinery is not demonstrated to help.

Not done: the dynamic-alpha decision table (spec section 8) is not implemented or tested; DC-SIS, elastic net and
group-aware screens are not implemented; D2-D9 scenarios, quantile-outcome and variance-only drift, and n x p x rho
grids beyond the above are not run. 5-10 replicates only.
