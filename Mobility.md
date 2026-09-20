# Mobility / CMA fine-tuning

This page documents the assistive-mobility object-detection jobs in the four-domain LEGO training collection. It is an engineering status and configuration guide, not a clinical validation or accessibility-performance claim.

As of **2026-09-20T16:32:04Z**, all **9 formal Mobility jobs** have completion markers and completed-run logs with no nonfinite loss reporting windows. This is log-level evidence, not a new tensor-level checkpoint or accuracy audit. See [the four-domain status](docs/training-status.md) and [machine-readable inventory](training/status.json).

## Data and label contract

The project's COCO-format Mobility split contains:

| Split | Images before training filters | Bounding-box annotations |
|---|---:|---:|
| Train | 8,456 | 21,850 |
| Validation | 784 | 1,580 |

The five classes are ordered **person, wheelchair, rollator, crutch, cane**. Source annotation category IDs are 0–4 in that same order; the detector must use the corresponding contiguous class-index mapping. Do not assume the one-based category IDs used in the other domains.

Training configs filter empty-ground-truth and undersized images. The observed dataloader has 961 steps per epoch at batch size 8, or 3,844 at batch size 2. Raw annotation image counts and training steps therefore should not be treated as interchangeable. The historical `mobility10k` prefix does not mean that these runs use exactly 10,000 training images.

This repository does not redistribute Mobility images or per-image annotations. Dataset access and usage permissions must be obtained separately. The split counts here describe the project export, not a claim about an official benchmark split.

## The nine configurations

Each link points to the corresponding published configuration. Schedules are intentionally model-specific; they are not a uniform-compute comparison.

| Detector / config | Schedule | Completed steps / schedule |
|---|---|---:|
| [FCOS R50](training/configs/mobility10k_ft_fcos_r50.py) | 12 epochs | 11,532 / 11,532 |
| [ATSS R50](training/configs/mobility10k_ft_atss_r50.py) | 12 epochs | 11,532 / 11,532 |
| [Dynamic R-CNN R50](training/configs/mobility10k_ft_dynamic_rcnn_r50.py) | 12 epochs | 11,532 / 11,532 |
| [RTMDet-Tiny](training/configs/mobility10k_ft_rtmdet_tiny.py) | 24 epochs | 23,064 / 23,064 |
| [YOLOF R50](training/configs/mobility10k_ft_yolof_r50.py) | 24 epochs | 23,064 / 23,064 |
| [SSD300](training/configs/mobility10k_ft_ssd300.py) | 24 epochs | 23,064 / 23,064 |
| [Deformable DETR R50](training/configs/mobility10k_ft_deformable_detr_r50.py) | 24 epochs | 92,256 / 92,256 |
| [Mask R-CNN Swin-T](training/configs/mobility10k_ft_mask_rcnn_swin_t.py) | 8 epochs | 30,752 / 30,752 |
| [DiffusionDet R50](training/configs/mobility10k_ft_diffusiondet_r50.py) | 50,000 iterations | 50,000 / 50,000 |

Completion was checked against final checkpoint-save and validation records, not just a `TRAIN_DONE` marker. For Deformable DETR, the preserved original 24-epoch run log is authoritative: a later accidental repeat stopped in epoch 1 and did not replace the completed checkpoints.

## Reproduction and reporting boundaries

Start from the linked model-specific configuration, preserve the class ordering, and supply your own permitted dataset and compatible pretrained weights. The published configuration collection is an archived experiment record, not a promise that every inherited donor recipe is equivalent or optimal.

Before reporting accuracy, bind the evaluated checkpoint and prediction files by digest, identify the exact validation split and resizing/postprocessing settings, and regenerate stale predictions after any checkpoint change. Earlier repeat/skip defects required regeneration of several Mobility prediction caches; a training-complete status alone does not certify that a particular cached prediction file is current.

No new mAP table, latency number, energy result, or deployment guarantee is asserted on this page. Cross-domain comparisons must retain each domain's different label set and split protocol.
