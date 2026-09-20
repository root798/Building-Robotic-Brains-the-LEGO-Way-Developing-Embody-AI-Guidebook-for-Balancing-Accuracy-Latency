# Training and evaluation

## Environment

The recipes were run with Python 3.8, PyTorch 1.10.2, MMCV 2.1.0 (compiled
operators), MMEngine 0.10.7, and MMDetection 3.3.0. This records the historical
environment; a complete installation lock is not included. Legacy transforms
and project modules require a compatible checkout.

Use an isolated environment. DiffusionDet, EfficientDet, and ViTDet require
their corresponding MMDetection `projects` modules. Configuration files retain
these imports. [Known issues](known-issues.md) describes recipe compatibility.

## Data and initialization

Run commands from the repository root. Obtain the data under its provider’s
terms and prepare COCO annotations with the [documented classes](datasets.md).

| Domain | Relative dataset root | Training annotation |
| --- | --- | --- |
| BDD100K-FT | `data/bdd100k_ft/` | `annotations/bdd10k_ft_train.json` |
| Mobility / CMA | `data/Dataset/mobility_coco/` | `annotations/instances_train.json` |
| SeaDronesSee | `data/Dataset/SeaDronesSeeObjectDetectionv2/mmdet_format/` | `annotations/instances_train.json` |
| WAID | `data/Dataset/WAID/WAID/` | `annotations_mmdet/instances_train.json` |

Place initialization weights in `checkpoints/`, matching each configuration’s
`load_from` and `init_cfg.checkpoint` fields. Fine-tuned checkpoint identities
are recorded separately in the [model catalog](../model_zoo/catalog.json).
Weights, images, and split-generation scripts are not bundled.

## Train one model

Choose an available GPU, then run a recipe with a new output directory:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -m training.train \
  model_zoo/configs/mobility/mobility10k_ft_atss_r50.py \
  --work-dir work_dirs/mobility_atss_run01
```

The launcher uses MMEngine, retains the SSD compatibility adapter, and requires
a fresh output directory. Training schedules and built-in validation come from
the configuration. The [campaign recipes](../training/configs/) also include
the separate BDD RepPoints v3 numerical repair.

## Evaluate a checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -m evaluation.evaluate \
  model_zoo/configs/mobility/mobility10k_ft_atss_r50.py \
  --checkpoint checkpoints/mobility_atss.pth \
  --annotations data/Dataset/mobility_coco/annotations/instances_val.json \
  --images data/Dataset/mobility_coco/images \
  --output work_dirs/mobility_atss_eval01
```

This produces COCO predictions and aggregate metrics with input-file hashes.
The [protocol](evaluation.md) matches the published native-scale evaluation.
Use the exact evaluated checkpoint to reproduce a catalog row. This command
measures accuracy, not inference latency.

## Numerical regression tests

In the compatible training environment, the loss-guard tests run on CPU:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m unittest discover -s tests -p 'test_finite_loss_hook.py' -v
```

The optional kernel regression needs an available GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.test_focal_cuda --run-cuda
```

See [numerical stability](numerical-stability.md) for the repair and acceptance checks.
