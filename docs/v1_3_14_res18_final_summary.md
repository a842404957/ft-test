# V1.3.14-final Res18 Summary

V1.3.14-final freezes the Res18 `ft_codebook_budgeted_translate` adapted artifact:

```text
results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/artifacts
```

The primary fault model is OU weight-level stuck-at with `stuck_at_zero` / `stuck_at_one` sampled at approximately `1:1`, `fault_rate=0.03`, and default Level1-only repair. Bit-flip remains a secondary stress test only.

## 10-Seed Stuck-at Main Table

| Seed | Baseline | Faulty | Level1 FT | Recovery | Correction | Repair Improved | Total Faults | Stuck 0/1 Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 92.40% | 10.00% | 91.40% | 98.79% | 95.78% | 96.47% | 41684 | 0.9967 |
| 43 | 92.40% | 10.00% | 75.42% | 79.39% | 95.61% | 96.38% | 41646 | 1.0006 |
| 44 | 92.40% | 10.00% | 89.43% | 96.40% | 95.85% | 96.45% | 41761 | 0.9991 |
| 45 | 92.40% | 10.00% | 91.34% | 98.71% | 95.77% | 96.67% | 41888 | 0.9989 |
| 46 | 92.40% | 10.00% | 91.05% | 98.36% | 95.69% | 96.52% | 41678 | 1.0013 |
| 47 | 92.40% | 10.00% | 90.38% | 97.55% | 95.85% | 96.52% | 41736 | 0.9962 |
| 48 | 92.40% | 10.00% | 91.03% | 98.34% | 95.77% | 96.51% | 41748 | 1.0012 |
| 49 | 92.40% | 10.00% | 89.67% | 96.69% | 95.78% | 96.46% | 41519 | 0.9954 |
| 50 | 92.40% | 10.00% | 90.66% | 97.89% | 95.70% | 96.56% | 42045 | 1.0020 |
| 51 | 92.40% | 10.00% | 90.19% | 97.32% | 95.82% | 96.46% | 41462 | 0.9942 |

Summary:

| Metric | Value |
| --- | ---: |
| seed_count | 10 |
| recovery_mean | 95.94% |
| recovery_std | 5.87% |
| recovery_min | 79.39% |
| recovery_p10 | 94.70% |
| recovery_median | 97.72% |
| recovery_max | 98.79% |
| worst_seed | 43 |
| median_seed | 47 |
| best_seed | 42 |
| repair_improved_mean | 96.50% |
| repair_improved_min | 96.38% |
| stuck_zero_one_ratio_mean | 0.9986 |

## Oracle Sanity

| Seed | Role | Baseline | Faulty | Oracle FT | Recovery |
| ---: | --- | ---: | ---: | ---: | ---: |
| 43 | worst | 92.40% | 10.00% | 92.40% | 100.00% |
| 47 | median | 92.40% | 10.00% | 92.40% | 100.00% |
| 42 | best | 92.40% | 10.00% | 92.40% | 100.00% |

Oracle restores baseline on the worst, median, and best seeds, so the stuck-at injection, detailed mask reuse, and restore pipeline are valid.

## Seed43 Layer Impact

Seed43 is the worst seed. Layer oracle was evaluated from the same exported detailed fault mask. `faulted_ous` and `whole_layer` give the same result because non-faulted weights in the Level1 repaired state are already the original weights.

| Layer | Faults | Level1 Corrected | Uncorrected | Residual Sum | Faulted-OU Oracle FT | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conv11.weight | 1998 | 1961 | 74 | 105.00 | 75.86% | +0.44% |
| conv12.weight | 1931 | 1434 | 994 | 170.94 | 75.54% | +0.12% |
| shortcut3.weight | 3927 | 3915 | 24 | 30.62 | 75.45% | +0.03% |
| conv16.weight | 7765 | 7753 | 24 | 49.18 | 75.35% | -0.07% |
| conv13.weight | 1975 | 1405 | 1140 | 183.15 | 75.33% | -0.09% |
| conv15.weight | 7795 | 7745 | 100 | 107.01 | 75.22% | -0.20% |
| conv14.weight | 3935 | 3891 | 88 | 112.47 | 75.20% | -0.22% |

Fault-exclusion uses the same seed43 fault mask and removes one target layer's faults without resampling to keep the same total fault count. Single-layer exclusions do not recover faulty accuracy above 10.00%, and Level1 marginal gains remain small.

## Gate Verdict

Primary stuck-at revised gate:

| Gate | Result |
| --- | --- |
| seeds >= 10 | pass |
| mean recovery >= 90% | pass, 95.94% |
| min recovery >= 75% | pass, 79.39% |
| FT accuracy >= faulty for every seed | pass |
| repair improved mean >= 95% | pass, 96.50% |
| stuck-at 0/1 ratio close to 1:1 | pass, mean 0.9986 |
| oracle restores baseline on worst/median/best | pass |

Res18 passes the revised primary gate.

## Why Not Continue Seed43 Patching

V1.3.13 `best_pair` and `weighted_average` reduced critical-layer residual error but did not materially improve seed43 accuracy. V1.3.14 layer oracle shows the largest single-layer gain is only `+0.44%` on `conv11.weight`, and fault-exclusion shows no single target layer explains the collapse. Seed43 is a distributed worst-case, not a single-layer repair-selection bug.

## Why Enter V2.0.0

The Res18 evidence package now demonstrates that codebook-aware adaptation plus default Level1 repair works under the target stuck-at 0/1=1:1 model across 10 seeds. The next project risk is cross-model transfer, not another Res18 seed43 patch. V2.0.0 should therefore start with a Vgg16 pilot, then proceed to a Res50 pilot if Vgg16 shows positive signal.
