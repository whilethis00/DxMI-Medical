# Source Layout

Reusable project code lives here. Scripts should stay thin and call into these modules where possible.

```text
src/
  data/        LIDC dataset and preprocessing-related utilities
  evaluation/  metrics and evaluation helpers
  models/      EBM, Flow Matching, and IRL modules
```

Current entrypoints still live in `scripts/` to preserve existing experiment commands.
