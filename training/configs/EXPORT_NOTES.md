# Campaign configurations

This directory contains the 20 completion-campaign recipes plus the separate
BDD RepPoints v3 candidate. The full collection is in
[model_zoo/configs](../../model_zoo/configs/).

The [manifest](manifest.json) records source/export hashes, validation splits,
custom imports, and initialization checkpoint filenames. Exports change storage
paths and the repair module import; model and optimization settings are retained.

## Layout

- Dataset paths are relative to `data/`.
- Initialization weights are referenced under `checkpoints/`.
- Outputs use `work_dirs/<model>/`.
- The nested dataloader dataset blocks determine the data consumed by each run.

See [setup](../../docs/reproduction.md) for dataset layouts and project-module
dependencies, and [known issues](../../docs/known-issues.md) for legacy settings.

## Maintainer export

```bash
python tools/export_training_configs.py \
  --source-root /path/to/experiment \
  --source-data-root /path/to/storage
```

Use `--check` to verify a deterministic export without writing files.
The exporter reads the declared queue and configurations, checks path-only
changes, and does not import training code.
