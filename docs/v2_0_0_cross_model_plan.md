# V2.0.0 Cross-Model Plan

V2.0.0 extends the validated Res18 stuck-at evidence to larger models. The order is fixed:

1. Vgg16 pilot
2. Res50 pilot

The primary fault model remains OU weight-level stuck-at 0/1 with approximately `1:1` ratio. Bit-flip is kept as secondary stress evidence and is not a hard gate.

## Vgg16 Pilot

Goals:

- Generate a `ft_codebook_budgeted_translate` adapted artifact.
- Run stuck-at 3% Level1-only full samples on seeds `42,43,44`.
- Run oracle on seed `42`.
- If the 3-seed pilot has positive signal, extend to 10 seeds.

Build and short adaptation:

```bash
python main.py \
  --model Vgg16 \
  --translate ft_codebook_budgeted_translate \
  --run-tag vgg16_codebook_adapt \
  --ft-grouping-mode codebook_budgeted \
  --ft-codebook-layer-config ft_codebook_layer_config_vgg16.json \
  --ft-mask-codebook-size 4 \
  --ft-mask-codebook-keep-counts 4,2 \
  --ft-mask-codebook-source mixed \
  --ft-mask-codebook-assign mixed \
  --ft-prototype-budget-ratio 0.25 \
  --ft-budget-target-coverage 0.6 \
  --ft-max-singleton-error 1.5 \
  --ft-force-prototype-assignment \
  --ft-normalize-prototype-vectors \
  --ft-prototype-space normalized_direction \
  --ft-codebook-adapt-epochs 10 \
  --ft-projection-ramp-start 0.0 \
  --ft-projection-ramp-end 0.1 \
  --ft-projection-ramp-epochs 151,160 \
  --ft-projection-loss-lambda 1e-4 \
  --ft-codebook-freeze-grouping
```

Three-seed stuck-at Level1-only pilot:

```bash
python scripts/run_fault_seed_sweep.py \
  --model Vgg16 \
  --translate ft_codebook_budgeted_translate \
  --config fault_tolerance_config_stuck_at_3pct.json \
  --repair-mode normal \
  --levels level1 \
  --samples -1 \
  --seeds 42,43,44 \
  --artifact-dir results/ft_runs/Vgg16/ft_codebook_budgeted_translate/vgg16_codebook_adapt/artifacts \
  --output-dir results/ft_runs/Vgg16/ft_codebook_budgeted_translate/vgg16_codebook_adapt/stuck_at_seed_sweep_3
```

Oracle seed42 sanity:

```bash
python run_hierarchical_fault_tolerance.py \
  --mode single \
  --model Vgg16 \
  --translate ft_codebook_budgeted_translate \
  --config fault_tolerance_config_stuck_at_3pct.json \
  --repair-mode oracle \
  --levels all \
  --samples -1 \
  --fault-seed 42 \
  --artifact-dir results/ft_runs/Vgg16/ft_codebook_budgeted_translate/vgg16_codebook_adapt/artifacts \
  --output-dir results/ft_runs/Vgg16/ft_codebook_budgeted_translate/vgg16_codebook_adapt/oracle_seed_42
```

If seed `42,43,44` show positive signal, extend to 10 seeds:

```bash
python scripts/run_fault_seed_sweep.py \
  --model Vgg16 \
  --translate ft_codebook_budgeted_translate \
  --config fault_tolerance_config_stuck_at_3pct.json \
  --repair-mode normal \
  --levels level1 \
  --samples -1 \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --artifact-dir results/ft_runs/Vgg16/ft_codebook_budgeted_translate/vgg16_codebook_adapt/artifacts \
  --output-dir results/ft_runs/Vgg16/ft_codebook_budgeted_translate/vgg16_codebook_adapt/stuck_at_seed_sweep_10
```

## Res50 Pilot

Res50 is deeper and should not start with adaptation directly. First run build-only projection sanity, then a short adaptation only if the projection sanity is usable.

Build-only projection sanity:

```bash
python main.py \
  --model Res50 \
  --translate ft_codebook_budgeted_translate \
  --build-only \
  --force-rebuild \
  --run-tag res50_codebook_build \
  --ft-grouping-mode codebook_budgeted \
  --ft-codebook-layer-config ft_codebook_layer_config_res50.json \
  --ft-mask-codebook-size 4 \
  --ft-mask-codebook-keep-counts 4,2 \
  --ft-mask-codebook-source mixed \
  --ft-mask-codebook-assign mixed \
  --ft-prototype-budget-ratio 0.25 \
  --ft-budget-target-coverage 0.6 \
  --ft-max-singleton-error 1.5 \
  --ft-force-prototype-assignment \
  --ft-normalize-prototype-vectors \
  --ft-prototype-space normalized_direction \
  --ft-projection-strength 0.05 \
  --ft-evaluate-projected
```

Short adaptation if projection sanity is acceptable:

```bash
python main.py \
  --model Res50 \
  --translate ft_codebook_budgeted_translate \
  --run-tag res50_codebook_adapt \
  --ft-grouping-mode codebook_budgeted \
  --ft-codebook-layer-config ft_codebook_layer_config_res50.json \
  --ft-mask-codebook-size 4 \
  --ft-mask-codebook-keep-counts 4,2 \
  --ft-mask-codebook-source mixed \
  --ft-mask-codebook-assign mixed \
  --ft-prototype-budget-ratio 0.25 \
  --ft-budget-target-coverage 0.6 \
  --ft-max-singleton-error 1.5 \
  --ft-force-prototype-assignment \
  --ft-normalize-prototype-vectors \
  --ft-prototype-space normalized_direction \
  --ft-codebook-adapt-epochs 10 \
  --ft-projection-ramp-start 0.0 \
  --ft-projection-ramp-end 0.1 \
  --ft-projection-ramp-epochs 151,160 \
  --ft-projection-loss-lambda 1e-4 \
  --ft-codebook-freeze-grouping
```

Seed42 stuck-at pilot:

```bash
python run_hierarchical_fault_tolerance.py \
  --mode single \
  --model Res50 \
  --translate ft_codebook_budgeted_translate \
  --config fault_tolerance_config_stuck_at_3pct.json \
  --repair-mode normal \
  --levels level1 \
  --samples -1 \
  --fault-seed 42 \
  --artifact-dir results/ft_runs/Res50/ft_codebook_budgeted_translate/res50_codebook_adapt/artifacts \
  --output-dir results/ft_runs/Res50/ft_codebook_budgeted_translate/res50_codebook_adapt/stuck_at_seed42
```

If seed42 is usable, expand to seeds `42,43,44`:

```bash
python scripts/run_fault_seed_sweep.py \
  --model Res50 \
  --translate ft_codebook_budgeted_translate \
  --config fault_tolerance_config_stuck_at_3pct.json \
  --repair-mode normal \
  --levels level1 \
  --samples -1 \
  --seeds 42,43,44 \
  --artifact-dir results/ft_runs/Res50/ft_codebook_budgeted_translate/res50_codebook_adapt/artifacts \
  --output-dir results/ft_runs/Res50/ft_codebook_budgeted_translate/res50_codebook_adapt/stuck_at_seed_sweep_3
```

## Gate

Vgg16 pilot should continue to 10 seeds only if:

- Oracle seed42 restores baseline.
- Level1-only FT accuracy is not lower than faulty accuracy for all three seeds.
- Recovery is positive and repair-improved rate remains high.

Res50 pilot should continue beyond seed42 only if:

- Build-only projection sanity does not collapse accuracy at low projection strength.
- Short adaptation produces a usable after-translate model.
- Seed42 Level1-only improves over faulty.
