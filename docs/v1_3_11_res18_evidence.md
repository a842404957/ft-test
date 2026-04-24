# V1.3.11 Res18 Evidence Package

V1.3.11 freezes the Res18 `ft_codebook_budgeted_translate` adapted artifact as the first positive
evidence package. The goal is validation and reporting, not another core grouping change.

## Current Valid Results

Artifact:

`results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/artifacts`

| run | samples | baseline | faulty | FT | recovery | correction | repair improved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Level1-only rerun | 1000 | 92.00% | 83.70% | 91.40% | 92.77% | 95.64% | 91.49% |
| Level1-only full | 10000 | 92.40% | 84.21% | 91.69% | 91.33% | 95.64% | 91.49% |
| Oracle rerun | 1000 | 92.00% | 83.70% | 92.00% | 100.00% | 100.00% | 95.56% |

The full-sample result confirms that the 1000-sample recovery was not a sampling artifact.

The older directories below are invalid and should not be used in V1.3.11 tables:

- `sim_stress_level1`
- `sim_stress_oracle`

They were produced before the save-time double-projection bug was fixed.

## Commands

Primary multi-seed Level1-only full runs:

```bash
for seed in 42 43 44; do
  python run_hierarchical_fault_tolerance.py \
    --mode single \
    --model Res18 \
    --translate ft_codebook_budgeted_translate \
    --config fault_tolerance_config_stress_3pct.json \
    --repair-mode normal \
    --levels level1 \
    --samples -1 \
    --fault-seed ${seed} \
    --artifact-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/artifacts \
    --output-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/sim_stress_3pct_level1_seed${seed}_full
done
```

Oracle sanity:

```bash
python run_hierarchical_fault_tolerance.py \
  --mode single \
  --model Res18 \
  --translate ft_codebook_budgeted_translate \
  --config fault_tolerance_config_stress_3pct.json \
  --repair-mode oracle \
  --levels all \
  --samples -1 \
  --fault-seed 42 \
  --artifact-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/artifacts \
  --output-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/sim_stress_3pct_oracle_seed42_full
```

Aggregate evidence reports:

```bash
python scripts/collect_ft_results.py \
  --aggregate-evidence-root results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt \
  --evidence-output-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/evidence
```

This writes:

- `evidence_summary.csv`
- `evidence_summary.json`
- `evidence_summary.md`

## Acceptance Gate

- `stress_3pct` Level1-only multi-seed average recovery rate >= 85%.
- No valid Level1-only seed below 80% recovery.
- Oracle full recovery should match baseline within 0.2%.
- Level1-only FT accuracy must be no worse than faulty accuracy.
- Old FTScore baseline should be run under the same `stress_3pct` Level1-only full setting before
  claiming codebook-adapt superiority.

## Next Direction

If the Res18 evidence gate passes, proceed to Vgg16 only with the same staged process:

1. build-only codebook structure diagnostics
2. projection sanity
3. short codebook-aware adaptation
4. Level1-only stress
5. oracle sanity
