# V1.3.10 Projection-Aware Adaptation Notes

## Why V1.3.9 Build-Only Failed

`V1.3.9` already showed that the `codebook_budgeted` route can build strong redundancy structure:

- `avg_repairable_ou_ratio = 0.8023`
- `avg_singleton_ratio = 0.3633`
- `avg_group_size = 3.6288`
- `assignment_error_p95 = 0.2755`

But the same run also showed that a one-shot projected model is not directly usable:

- `projected_accuracy ≈ 0.14`
- `projected_accuracy_drop ≈ 0.8070`

So the failure point is not the grouping structure itself. The failure point is applying a strong projection to an unadapted checkpoint and evaluating it immediately.

## Why This Is Different From PRAP

PRAP does not stop at “build pruning/mapping relation, then evaluate a hard-translated model”. Its effective flow is:

1. build structural constraint / reuse relation
2. run aware training with regularization
3. periodically translate/evaluate during training

That means the model is gradually adapted to the imposed structure.

Our `V1.3.9` build-only path skipped step 2. It effectively tested:

`original checkpoint -> hard projection -> direct evaluation`

That is a much harsher condition than the PRAP-style “build first, adapt later” process.

## Why V1.3.10 Introduces Short Codebook-Aware Training

`V1.3.10` keeps the `V1.3.9` codebook/group structure, but changes the validation procedure:

1. first run projection sanity at small `projection_strength`
2. then run a short codebook-aware adaptation stage
3. use a projection ramp instead of a single strong projection
4. keep grouping frozen by default

The goal is not to prove full final performance yet. The goal is to answer a narrower question:

> If the structure is already good, can a short adaptation stage make the projected model usable?

## Why The Old Build-Only Gate Was Too Strict

For `V1.3.9`, the gate `projected_accuracy_drop <= 5%` was reasonable as a sanity check, but too strict as the only go/no-go condition for a freshly introduced structural method. A new structure may be valid while still requiring a short adaptation stage.

So `V1.3.10` splits the decision into two gates:

1. **Projection sanity gate**
   Small projection strengths such as `0.05` or `0.1` should not collapse completely.
2. **Adaptation gate**
   After short aware training, `after_translate_accuracy_drop` should fall into an acceptable first-pass range and be clearly better than raw build-only projection.

## What V1.3.10 Is Not

`V1.3.10` is **not**:

- a return to PRAP as the main path
- a full long FT training pass
- a simulator-first validation phase
- another round of structural parameter sweep

It is a controlled bridge between:

- `V1.3.9`: strong structure, unusable projection
- future stage: structure-preserving, projection-adapted model ready for `Level1-only` stress

## Res18 Projection Sanity Result

The first sanity sweep reused the same `res18_codebook_budget_build` artifacts and only changed
`projection_strength`.

| projection_strength | projected_accuracy | projected_accuracy_drop |
| ---: | ---: | ---: |
| 0.00 | 0.9063 | 0.0407 |
| 0.05 | 0.9044 | 0.0426 |
| 0.10 | 0.8928 | 0.0542 |
| 0.25 | 0.6874 | 0.2596 |

This indicates that the structure is not inherently unusable. Small projection strengths around
`0.05` to `0.10` remain within a reasonable first-pass accuracy range, while `0.25` is already too
aggressive for an unadapted checkpoint.

## Res18 Short Adaptation Result

The first codebook-aware short adaptation run used:

```bash
python main.py \
  --model Res18 \
  --translate ft_codebook_budgeted_translate \
  --artifact-dir results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/artifacts \
  --ft-grouping-mode codebook_budgeted \
  --ft-codebook-adapt-epochs 10 \
  --ft-projection-ramp-start 0.0 \
  --ft-projection-ramp-end 0.1 \
  --ft-projection-ramp-epochs 151,160 \
  --ft-projection-loss-lambda 1e-4 \
  --ft-projection-reg-max-links 8192 \
  --ft-codebook-freeze-grouping \
  --no-ft-codebook-use-legacy-regularization \
  --ft-reg-interval 10
```

The important implementation detail is `--ft-projection-reg-max-links 8192`. Without link sampling,
projection consistency scans close to one million member links per regularized batch and is too slow
for iteration. The sampled run recorded about `8140 / 976729` projection links per regularized batch.

Final result:

| metric | value |
| --- | ---: |
| train_accuracy | 99.024% |
| test_accuracy before translate | 93.39% |
| test_accuracy after translate | 92.27% |
| translate accuracy difference | 1.12% |
| training time | 13m 58s |
| final coverage | 0.9589 |
| final group_count | 266215 |
| final singleton_ratio | 0.2127 |
| final exact_group_proportion | 0.0015 |
| final scaled_group_proportion | 0.7857 |

The adaptation gate passes for this first prototype:

- `after_translate_accuracy_drop = 1.12%`, below the first-pass 10% threshold.
- The model remains usable after projection, unlike the V1.3.9 one-shot `projection_strength=0.5`
  run.
- The structure remains strong enough for a Level 1 focused stress test.

One caveat: `projection_metrics.json` may still reflect the previous build-only projection sanity
run if build-only was executed after training. For the adapted model, use the training log,
`training_profile.csv`, and `after_translate_parameters.pth` as the source of truth.

## Next Gate

The next validation step should be `Level1-only` stress on the adapted artifacts, with oracle run as
the pipeline sanity check. Level 2/3 correction counts should still not be used as the main evidence.

## Fault-Tolerance Baseline Caveat

The first `Level1-only` and `oracle` stress runs on `res18_codebook_adapt` reported a simulator
baseline accuracy of `85.74%`, which is inconsistent with the adaptation log
`After_translate_accuracy = 92.27%`.

The root cause is a save-time bug in `ft_group_translate_train`:

1. epoch `160` applied projection with `projection_strength = 0.1`
2. the code evaluated and printed `After_translate_accuracy = 92.27%`
3. after the training loop, the final save path unconditionally applied projection again
4. `model_Res18_ft_codebook_budgeted_translate_after_translate_parameters.pth` therefore contained
   a twice-projected model

Direct reevaluation of the saved model confirmed the mismatch:

| evaluation subset | accuracy |
| --- | ---: |
| first 1024 test images | 85.74% |
| full 10000 test images | 86.19% |

The simulator was loading the twice-projected file correctly; the artifact itself was wrong. The fix
is to skip the final projection when the final translate epoch already applied the same
`projection_strength`. Existing `sim_stress_level1` and `sim_stress_oracle` reports from this
artifact should be considered invalid and rerun after regenerating the adapted artifact.

## Fixed Rerun Result

After applying the save-time fix, the same short adaptation command regenerated the artifact at
`results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/artifacts`.

The corrected adaptation record is:

| metric | value |
| --- | ---: |
| before translate accuracy | 93.63% |
| after translate accuracy | 92.40% |
| translate accuracy difference | 1.23% |
| saved model accuracy, first 1000 test images | 92.00% |
| saved model accuracy, full 10000 test images | 92.40% |

This confirms that `after_translate_parameters.pth` now matches the printed `After_translate`
accuracy instead of containing a twice-projected model.

The corrected `stress_3pct` rerun results are:

| run | baseline | faulty | ft | recovery | correction rate | repair improved rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Level1-only | 92.00% | 83.70% | 91.40% | 92.77% | 95.64% | 91.49% |
| Oracle | 92.00% | 83.70% | 92.00% | 100.00% | 100.00% | 95.56% |

Rerun report paths:

- `results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/sim_stress_level1_rerun/fault_tolerance_report_20260424_133344.md`
- `results/ft_runs/Res18/ft_codebook_budgeted_translate/res18_codebook_adapt/sim_stress_oracle_rerun/fault_tolerance_report_20260424_133720.md`

Interpretation:

- The oracle run confirms the fault injection / restoration / evaluation pipeline is valid.
- Level1-only is now a real positive result: it recovers `7.70%` absolute accuracy out of the
  `8.30%` stress-induced drop on the first 1000 samples.
- The remaining `0.60%` gap to baseline is consistent with non-repairable or not-effectively-improved
  Level 1 repairs; it should be investigated per layer before claiming full robustness.
