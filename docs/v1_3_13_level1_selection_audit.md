# V1.3.13 Level1 Selection Audit

## Current V1.3.12 Level1 behavior

Level 1 repairs each faulty OU inside its redundancy group.

- If the group has a prototype block and the aligned prototype OU is healthy, it is selected first.
- If the prototype is unavailable, the healthy group member with the highest cosine similarity to the faulty OU's healthy reference weight is selected.
- The replacement scale is computed as `faulty_multiplier / replacement_multiplier`.
- Zero, NaN, and infinite scale factors are rejected and counted as `level1_zero_scale_failed`.
- Repair quality is recorded after applying the replacement, but the selection step does not use precomputed expected repair error.

## Seed43 failure interpretation

V1.3.12 seed43 diagnostics show that Level2/3 increase correction count but do not improve accuracy. The issue is therefore not just uncorrected count.

- `conv12.weight` and `conv13.weight` have higher uncorrected counts, so coverage remains insufficient.
- `conv14.weight`, `conv15.weight`, and `conv11.weight` have high Level1 correction counts but dominate residual error, so candidate quality matters.
- Prototype-first selection can choose a healthy OU whose group relation is valid structurally but not the lowest-error repair candidate for a sensitive layer.

## V1.3.13 patch rationale

V1.3.13 adds an optional, block-aware Level1 repair cache. The cache is computed from the healthy adapted model before fault injection and is only used when explicitly requested. It is metadata, not oracle repair: runtime repair does not recompute the best candidate from fault outcomes.

The default path remains unchanged when `--level1-selection` is omitted.
