# LEGO-AI · Multi-Domain Detection Model Zoo

**Building Robotic Brains the LEGO Way** — detector recipes and evaluation
results for driving, mobility assistance, maritime rescue, and wildlife monitoring.
Maintained by [root798](https://github.com/root798).

## Explore the models

The collection covers **32 detector architectures across four domains**.
Each domain page includes AP, AP50, AP75, configuration links, and run status.

| Domain | Setting | Classes | Validation images | Model zoo |
| --- | --- | ---: | ---: | --- |
| BDD100K-FT | Driving | 11 | 4,000 | [32 architecture slots](BDD20K_FT.md) |
| Mobility / CMA | Mobility assistance | 5 | 784 | [32 architecture slots](Mobility.md) |
| SeaDronesSee | Maritime aerial detection | 5 | 1,547 | [32 architecture slots](SeaDronesSee.md) |
| WAID | Wildlife aerial detection | 6 | 2,873 | [32 architecture slots](WAID_FT.md) |

The [machine-readable catalog](model_zoo/catalog.json) contains **136 evaluation
records** plus two ongoing recipes, including earlier and replacement runs.
The four-domain main tables contain 127 evaluated slots and one training slot
(WAID ViTDet-B). BDD RepPoints v2 is marked for a numerical issue; its v3
replacement is tracked separately. Snapshot dates appear on each page.

## Use the collection

- [Results CSV](model_zoo/results.csv) — aggregate accuracy for every recorded run.
- [Model configurations](model_zoo/configs/) — portable recipes for all evaluated models.
- [Evaluation protocol](docs/evaluation.md) — splits, class mapping, scales, and metrics.
- [Reproduction guide](docs/reproduction.md) — environment, data layout, training, evaluation.
- [Fine-tuning progress](docs/training-status.md) — the 20-job completion campaign.
- [Numerical stability](docs/numerical-stability.md) — FP32 focal-loss adapter and tests.

Models are fine-tuned independently for each domain. AP is reported on the
project validation splits at each recipe’s native test scale. The catalog
records checkpoint SHA-256 identities; **weight downloads are not yet hosted**.
Images and annotations are obtained separately from the dataset providers.

## Quick start

With a compatible MMDetection environment and the required dataset and weights:

```bash
python -m training.train \
  model_zoo/configs/mobility/mobility10k_ft_atss_r50.py \
  --work-dir work_dirs/mobility_atss_run01

python -m evaluation.evaluate \
  model_zoo/configs/mobility/mobility10k_ft_atss_r50.py \
  --checkpoint checkpoints/mobility_atss.pth \
  --annotations data/Dataset/mobility_coco/annotations/instances_val.json \
  --images data/Dataset/mobility_coco/images \
  --output work_dirs/mobility_atss_eval01
```

See the [setup guide](docs/reproduction.md) for dependencies and initialization
weights. The example checkpoint is a local file supplied by the user.

## Development

Lightweight checks run without a GPU or ML installation:

```bash
python -m unittest discover -s tests -p 'test_training_configs.py'
python -m unittest discover -s tests -p 'test_publication.py'
python -m unittest discover -s tests -p 'test_model_zoo.py'
python -m unittest discover -s tests -p 'test_train_cli.py'
```

Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md).
Dataset references are in [docs/datasets.md](docs/datasets.md);
framework attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Recipe-specific limitations are collected in [known issues](docs/known-issues.md).
