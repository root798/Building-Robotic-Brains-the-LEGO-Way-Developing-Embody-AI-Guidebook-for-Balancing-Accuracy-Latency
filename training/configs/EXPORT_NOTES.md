# Training configuration snapshots

This directory contains the 20 formal fine-tuning queue recipes plus a separate
BDD RepPoints v3 numerical-repair candidate. The candidate is not a 21st distinct
model in the formal queue, and publishing it does not certify its final outcome.
See `manifest.json` for roles, validation split names, source SHA-256 values,
published SHA-256 values and external imports. Source files are identified by
basename only; no host paths or checkpoints are published.

These are historical recipe snapshots with portable paths, **not a one-command
reproduction environment**. Hyperparameters, samplers, augmentation,
normalization and framework-specific settings have not been silently repaired.
In particular, some historical pipelines include `Normalize` as well as a model
`DetDataPreprocessor`, and some omit explicit dataloader samplers. Reproducing
their exact behavior requires the corresponding framework versions and
registries. No hidden compatibility patch is assumed or supplied here. Changing
these settings is a new experiment, not a faithful replay.

## Relative storage layout

Run from the repository root. The common private storage prefix is replaced with
`data/`; the suffix and annotation names are retained:

| Domain | Actual dataloader dataset root |
| --- | --- |
| BDD | `data/bdd100k_ft/` |
| Mobility | `data/Dataset/mobility_coco/` |
| SeaDronesSee | `data/Dataset/SeaDronesSeeObjectDetectionv2/mmdet_format/` |
| WAID | `data/Dataset/WAID/WAID/` |

Dataset images and COCO-format annotations must be obtained/prepared separately
under the applicable dataset terms. Some inherited top-level `data_root` and
`_common_ds_kw` values still refer to the BDD donor; the nested
`train_dataloader.dataset`, `val_dataloader.dataset` and `test_dataloader.dataset`
are authoritative for the configured data consumed by each job. They are retained
as evidence rather than rewritten to look more uniform.

Local initialization checkpoints are referenced as `checkpoints/<basename>`.
Each manifest entry lists its local checkpoint references. Framework checkpoint
URIs such as `torchvision://...` and `open-mmlab://...` are unchanged. Outputs go
to `work_dirs/<model>`. Do not treat a validation-annotation `test_dataloader` in
a training snapshot as proof of a held-out test evaluation: follow the recorded
evaluation split protocol separately.

## External dependencies

The configurations use PyTorch, MMEngine, MMCV and MMDetection registries and
compiled detection operators. Additional upstream MMDetection project modules
must be available on the Python import path for these architectures:

| Recipes | Required external module |
| --- | --- |
| Mobility DiffusionDet | `projects.DiffusionDet.diffusiondet` |
| BDD EfficientDet-D3 | `projects.EfficientDet.efficientdet` |
| SeaDronesSee and WAID ViTDet-B | `projects.ViTDet.vitdet` |

Their third-party source, environments and weights are intentionally not copied.
The BDD RepPoints v3 candidate additionally imports the project-owned
`training.reppoints_numerics` module. It inherits the adjacent v2 configuration
and changes classification-loss precision while retaining the original training
recipe and a distinct output identity.

## Export and verification

The exporter requires explicit local paths as command-line arguments and reads
only the queue declaration plus its reviewed configuration allowlist:

```bash
python tools/export_training_configs.py \
  --source-root /path/to/private/experiment \
  --source-data-root /path/to/private/storage
python -m unittest discover -s tests -p test_training_configs.py
```

Add `--check` to compare a deterministic re-export with these published files
without writing anything. The tool rejects changes to the formal queue,
unrecognized absolute paths and surviving private host paths. It only adapts
designated string literals, checks that the remaining AST is unchanged, and
never imports a model or starts training. Public hashes verify publication
integrity, not training success, metric validity or checkpoint availability.
