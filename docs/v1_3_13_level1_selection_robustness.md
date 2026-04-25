# V1.3.13 Level1 Repair Selection Robustness

## Purpose

V1.3.13 keeps the Res18 adapted artifact fixed and only changes optional Level1 candidate selection. It does not retrain, rebuild grouping, change stuck-at injection, or use Level2/3 as the main evidence.

## New options

- `--level1-selection {default,best_pair,weighted_average}`
- `--level1-topk`
- `--level1-max-expected-error`
- `--level1-min-expected-improvement`
- `--level1-critical-layer-config`
- `--level1-cache-max-group-size`
- `--level1-cache-critical-layers-only`

Default behavior is unchanged. If `--level1-selection` is omitted, the simulator does not build the repair cache.

## Critical layer configs

- `level1_critical_layers_res18_bestpair.json`: `conv11/12/13/14/15` use `best_pair topk=3`.
- `level1_critical_layers_res18_weighted.json`: `conv11/12/13` use `best_pair topk=3`; `conv14/15` use `weighted_average topk=3`.

## Validation order

1. Run stuck-at seed43 with best_pair.
2. Run stuck-at seed43 with weighted config.
3. Use the better strategy to rerun stuck-at seed42/44.
4. Run bit-flip seed43 as a supplemental worst-case check.
5. Summarize stuck-at 42/43/44 and bit-flip seed43.

Gate: stuck-at seed43 recovery should reach at least 80%, seed42/44 should not materially regress, and repair improved rate should remain near the V1.3.12 stuck-at level.
