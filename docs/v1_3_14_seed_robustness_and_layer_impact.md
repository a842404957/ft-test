# V1.3.14 Seed Robustness And Layer Impact Diagnosis

V1.3.14 freezes the Res18 `ft_codebook_budgeted_translate` adapted artifact and does not change grouping, training, projection, or the stuck-at fault model. The main evidence uses default Level1 repair behavior. `best_pair` and `weighted_average` remain optional appendix checks only.

## Evidence Protocol

Fixed artifact:

```bash
results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/artifacts
```

First rerun the V1.3.12 stuck-at seeds `42,43,44` with default Level1 full samples and compare with the recorded values. If the values differ, record whether the difference comes from config, artifact, sample count, or script changes before using the 10-seed evidence.

Primary 10-seed stuck-at sweep:

```bash
python scripts/run_fault_seed_sweep.py \
  --model Res18 \
  --translate ft_codebook_budgeted_translate \
  --config fault_tolerance_config_stuck_at_3pct.json \
  --repair-mode normal \
  --levels level1 \
  --samples -1 \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --artifact-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/artifacts \
  --output-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/stuck_at_seed_sweep_10
```

The summary reports mean/std/min/max, median, q25/q75, p10 recovery, worst/median/best seed, FT >= faulty for every seed, repair-improved mean/min, and stuck-at zero/one ratio by affected weight count.

## Oracle Sanity

Run oracle on the selected worst/median/best seeds from `seed_sweep_summary.json`: worst seed `43`, median seed `47`, best seed `42`.

```bash
python run_hierarchical_fault_tolerance.py \
  --mode single \
  --model Res18 \
  --translate ft_codebook_budgeted_translate \
  --config fault_tolerance_config_stuck_at_3pct.json \
  --repair-mode oracle \
  --levels all \
  --samples -1 \
  --fault-seed 43 \
  --artifact-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/artifacts \
  --output-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/oracle_seed_43

python run_hierarchical_fault_tolerance.py \
  --mode single \
  --model Res18 \
  --translate ft_codebook_budgeted_translate \
  --config fault_tolerance_config_stuck_at_3pct.json \
  --repair-mode oracle \
  --levels all \
  --samples -1 \
  --fault-seed 47 \
  --artifact-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/artifacts \
  --output-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/oracle_seed_47

python run_hierarchical_fault_tolerance.py \
  --mode single \
  --model Res18 \
  --translate ft_codebook_budgeted_translate \
  --config fault_tolerance_config_stuck_at_3pct.json \
  --repair-mode oracle \
  --levels all \
  --samples -1 \
  --fault-seed 42 \
  --artifact-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/artifacts \
  --output-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/oracle_seed_42
```

Oracle must restore baseline on all three seeds. If it does not, the fault/restore pipeline is invalid for that seed.

## Seed43 Layer Diagnosis

Export seed43's detailed fault mask once and reuse it for all layer-impact and fault-exclusion analysis:

```bash
python scripts/analyse_layer_impact_seed.py \
  --model Res18 \
  --translate ft_codebook_budgeted_translate \
  --config fault_tolerance_config_stuck_at_3pct.json \
  --samples -1 \
  --fault-seed 43 \
  --artifact-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/artifacts \
  --output-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/seed43_layer_impact \
  --mode both \
  --export-fault-mask \
  --fault-mask-file results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/seed43_layer_impact/seed43_fault_mask.json
```

To rerun from the exact same mask:

```bash
python scripts/analyse_layer_impact_seed.py \
  --model Res18 \
  --translate ft_codebook_budgeted_translate \
  --config fault_tolerance_config_stuck_at_3pct.json \
  --samples -1 \
  --fault-seed 43 \
  --artifact-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/artifacts \
  --output-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/seed43_layer_impact_reuse \
  --mode both \
  --reuse-fault-mask \
  --fault-mask-file results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/seed43_layer_impact/seed43_fault_mask.json
```

Layer oracle reports both `faulted_ous` and `whole_layer`. The primary conclusion uses `faulted_ous`; `whole_layer` is only an upper bound.

Fault-exclusion removes the selected layer's faults from the same exported mask. It does not resample faults to keep the same total fault count, so each row reports `original_total_faults`, `remaining_total_faults`, `removed_faults`, and `removed_fault_ratio`.

## Revised Res18 To Vgg16 Gate

Primary stuck-at gate:

- `seeds >= 10`
- `mean recovery >= 90%`
- `min recovery >= 75%`
- `FT accuracy >= faulty accuracy` for every seed
- `repair improved mean >= 95%`
- `stuck_at_zero / stuck_at_one` ratio close to `1:1` by affected weight count
- Oracle restores baseline on worst, median, and best seeds

Bit-flip remains a secondary stress test and is not a hard gate for entering the Vgg16 pilot.

## Interpretation

If seed43 remains the worst seed but the 10-seed gate passes, proceed to Vgg16 pilot planning. If the gate fails, use `seed_layer_impact.*` and `seed_fault_exclusion.*` to distinguish coverage limits, replacement-quality limits, and sensitive-layer fault placement.

## Completed Res18 Results

The stuck-at 10-seed Level1-only full sweep passed the revised primary gate:

| Metric | Value |
| --- | ---: |
| seed_count | 10 |
| recovery_mean | 95.94% |
| recovery_std | 5.87% |
| recovery_min | 79.39% |
| recovery_p10 | 94.70% |
| recovery_q25 | 96.85% |
| recovery_median | 97.72% |
| recovery_q75 | 98.36% |
| recovery_max | 98.79% |
| worst_seed | 43 |
| median_seed | 47 |
| best_seed | 42 |
| repair_improved_mean | 96.50% |
| repair_improved_min | 96.38% |
| stuck_zero_one_ratio_mean | 0.9986 |

Oracle sanity passed on worst, median, and best seeds:

| Seed | Role | Baseline | Faulty | Oracle FT | Recovery |
| ---: | --- | ---: | ---: | ---: | ---: |
| 43 | worst | 92.40% | 10.00% | 92.40% | 100.00% |
| 47 | median | 92.40% | 10.00% | 92.40% | 100.00% |
| 42 | best | 92.40% | 10.00% | 92.40% | 100.00% |

Seed43 layer-impact oracle results use the same exported detailed fault mask. `faulted_ous` and `whole_layer` produced the same accuracy gains because non-faulted weights in the repaired state are already the original weights.

| Layer | Faults | Level1 Corrected | Uncorrected | Residual Sum | Layer Oracle FT | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conv11.weight | 1998 | 1961 | 74 | 105.00 | 75.86% | +0.44% |
| conv12.weight | 1931 | 1434 | 994 | 170.94 | 75.54% | +0.12% |
| shortcut3.weight | 3927 | 3915 | 24 | 30.62 | 75.45% | +0.03% |
| conv16.weight | 7765 | 7753 | 24 | 49.18 | 75.35% | -0.07% |
| conv13.weight | 1975 | 1405 | 1140 | 183.15 | 75.33% | -0.09% |
| conv15.weight | 7795 | 7745 | 100 | 107.01 | 75.22% | -0.20% |
| conv14.weight | 3935 | 3891 | 88 | 112.47 | 75.20% | -0.22% |

Fault-exclusion uses the same seed43 mask and removes faults from the selected layer without resampling to keep the total fault count. Faulty accuracy remains 10.00% for every single-layer exclusion, while FT marginal gain mirrors the layer-oracle result.

Conclusion: seed43 is the worst seed but still passes the revised min-recovery gate. Its remaining gap is not dominated by one target layer. `conv12/conv13` have clear coverage limits, while `conv11/14/15` carry large residual error, but single-layer upper bounds are too small to justify another Level1 selection patch. The Res18 stuck-at evidence supports moving to a Vgg16 pilot, with bit-flip kept as secondary stress evidence rather than a hard gate.
