# V1.3.12 Stuck-at Fault Migration and Multi-Seed Robustness

V1.3.12 fixes the evaluation target before moving to Vgg16. The Res18
`ft_codebook_budgeted_translate` adapted artifact remains fixed:

`results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/artifacts`

## Current Status

The V1.3.11 bit-flip multi-seed evidence is positive but not yet stable enough:

| seed | baseline | faulty | FT | recovery | correction | repair improved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 92.40% | 84.21% | 91.69% | 91.33% | 95.64% | 91.49% |
| 43 | 92.40% | 35.68% | 77.79% | 74.24% | 95.74% | 91.13% |
| 44 | 92.40% | 70.84% | 89.72% | 87.57% | 95.62% | 91.23% |

Average recovery is about 84.38%. The direction is valid, but the Res18 gate is not fully
passed because seed 43 falls below 80% recovery.

## New Interfaces

Weight-level stuck-at faults are now configured with:

- `fault_models`: `["stuck_at_zero", "stuck_at_one"]`
- `fault_model_probs`: usually `[0.5, 0.5]`
- `fault_weight_ratio`: fraction of weights inside each faulty OU to replace
- `fault_granularity`: `ou_weight`
- `stuck_at_one_value_mode`: `layer_absmax`, `constant_one`, or `global_absmax`
- `stuck_at_one_value`: optional explicit override
- `stuck_at_zero_value`: default `0.0`

The simulator records the exact affected weight indices, fault model, and replacement value in
`detailed_fault_mask`, and reuses that mask for FT/oracle evaluation.

## Commands

Bit-flip Level1-only seed sweep:

```bash
python scripts/run_fault_seed_sweep.py \
  --model Res18 \
  --translate ft_codebook_budgeted_translate \
  --config fault_tolerance_config_stress_3pct.json \
  --repair-mode normal \
  --levels level1 \
  --samples -1 \
  --seeds 42,43,44,45,46 \
  --artifact-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/artifacts \
  --output-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/bitflip_seed_sweep_3pct
```

Seed 43 diagnosis:

```bash
python scripts/analyse_fault_seed_failure.py \
  --evidence-root results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/bitflip_seed_sweep_3pct \
  --focus-seed 43 \
  --output-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/bitflip_seed43_diagnosis
```

Bit-flip oracle seed 43/44 sanity:

```bash
for seed in 43 44; do
  python run_hierarchical_fault_tolerance.py \
    --mode single \
    --model Res18 \
    --translate ft_codebook_budgeted_translate \
    --config fault_tolerance_config_stress_3pct.json \
    --repair-mode oracle \
    --levels all \
    --samples -1 \
    --fault-seed ${seed} \
    --artifact-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/artifacts \
    --output-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/sim_stress_3pct_oracle_seed${seed}_full
done
```

Stuck-at Level1-only first pass:

```bash
python scripts/run_fault_seed_sweep.py \
  --model Res18 \
  --translate ft_codebook_budgeted_translate \
  --config fault_tolerance_config_stuck_at_3pct.json \
  --repair-mode normal \
  --levels level1 \
  --samples -1 \
  --seeds 42,43,44 \
  --artifact-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/artifacts \
  --output-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/stuck_at_seed_sweep_3pct
```

Stuck-at oracle sanity:

```bash
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
  --output-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/stuck_at_oracle_seed42
```

## Vgg16 Gate

Do not move to Vgg16 until Res18 passes the evidence gate:

- average Level1-only recovery is at least 85%
- no seed is below 80% recovery
- FT accuracy is never below faulty accuracy
- repair improved rate stays at least 90%
- stuck-at oracle restores accuracy to baseline
