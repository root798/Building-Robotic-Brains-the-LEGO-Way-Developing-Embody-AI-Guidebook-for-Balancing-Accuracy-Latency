# BDD100K-FT model zoo

[All domains](README.md) · [CSV](model_zoo/results.csv) · [Protocol](docs/evaluation.md) · [Training](docs/training-status.md)

Snapshot: 2026-09-20T17:04:05+00:00. AP values use a 0–100 scale.

| Train images | Validation images | Classes | Architecture slots |
| ---: | ---: | ---: | ---: |
| 12,000 | 4,000 | 11 | 32 |

Classes: `person`, `rider`, `car`, `bus`, `truck`, `bike`, `motor`, `traffic light`, `traffic sign`, `train`, `animal`.

This is the project’s 11-class, 4,000-image validation split of BDD100K-FT.

## Models

Each row links to its recipe. Evaluated checkpoint filenames and SHA-256 identities are in the [catalog](model_zoo/catalog.json). Weight downloads are not hosted in this repository.

| Model | AP | AP50 | AP75 | Status |
| --- | ---: | ---: | ---: | --- |
| [ATSS R50](model_zoo/configs/bdd/bdd10k_ft_atss_r50.py) | 37.11 | 62.00 | 37.56 | Evaluated |
| [Cascade R-CNN R50](model_zoo/configs/bdd/bdd10k_ft_cascade_rcnn_r50.py) | 27.85 | 50.74 | 25.97 | Evaluated |
| [CenterNet R18-DCN](model_zoo/configs/bdd/bdd10k_ft_centernet_r18_dcn.py) | 3.99 | 9.92 | 2.87 | Evaluated |
| [Conditional DETR R50](model_zoo/configs/bdd/bdd20k_ft_conditional_detr_r50.py) | 22.57 | 45.31 | 19.36 | Evaluated |
| [DDQ-DETR R50](model_zoo/configs/bdd/bdd10k_ft_ddq_detr4scale_r50.py) | 33.89 | 59.53 | 32.31 | Evaluated |
| [Deformable DETR R50](model_zoo/configs/bdd/bdd10k_ft_deformable_detr_r50.py) | 27.83 | 52.30 | 25.53 | Evaluated |
| [Deformable DETR Refine R50](model_zoo/configs/bdd/bdd10k_ft_deformable_detr_refine_r50.py) | 28.93 | 52.77 | 27.43 | Evaluated |
| [DETR R50](model_zoo/configs/bdd/bdd20k_ft_detr_r50.py) | 18.19 | 37.40 | 15.21 | Evaluated |
| [DiffusionDet R50](model_zoo/configs/bdd/bdd10k_ft_diffusiondet_r50.py) | 29.44 | 55.35 | 26.74 | Checkpoint unavailable |
| [DINO Swin-L](model_zoo/configs/bdd/bdd10k_ft_dino_swin_l.py) | 32.16 | 59.07 | 29.70 | Evaluated |
| [Dynamic R-CNN R50](model_zoo/configs/bdd/bdd10k_ft_dynamic_rcnn_r50.py) | 25.56 | 46.41 | 24.31 | Evaluated |
| [EfficientDet-D3 v2](model_zoo/configs/bdd/bdd10k_ft_efficientdet_d3_v2.py) | 19.41 | 37.73 | 16.86 | Evaluated |
| [FCOS R50](model_zoo/configs/bdd/bdd10k_ft_fcos_r50.py) | 34.33 | 58.95 | 33.62 | Evaluated |
| [Faster R-CNN R50-FPN](model_zoo/configs/bdd/bdd10k_ft_frcnn_r50_fpn.py) | 32.07 | 57.71 | 30.97 | Evaluated |
| [FreeAnchor R50](model_zoo/configs/bdd/bdd10k_ft_freeanchor_r50.py) | 22.37 | 42.28 | 20.42 | Evaluated |
| [Grid R-CNN R50](model_zoo/configs/bdd/bdd20k_ft_grid_rcnn_r50.py) | 28.96 | 52.15 | 27.88 | Evaluated |
| [Mask R-CNN Swin-T](model_zoo/configs/bdd/bdd10k_ft_mask_rcnn_swin_t.py) | 21.95 | 45.05 | 18.20 | Evaluated |
| [NAS-FCOS R50](model_zoo/configs/bdd/bdd20k_ft_nas_fcos_r50.py) | 27.14 | 50.64 | 24.61 | Evaluated |
| [PAA R50](model_zoo/configs/bdd/bdd20k_ft_paa_r50.py) | 29.02 | 52.11 | 27.41 | Evaluated |
| [RepPoints R50 v2](model_zoo/configs/bdd/bdd10k_ft_reppoints_r50_v2.py) | 17.88 | 35.05 | 15.75 | NaN in training |
| [RetinaNet EfficientNet-B3](model_zoo/configs/bdd/bdd10k_ft_retinanet_effb3.py) | 23.79 | 43.73 | 22.25 | Evaluated |
| [RetinaNet PVT-T](model_zoo/configs/bdd/bdd10k_ft_retinanet_pvtt.py) | 24.45 | 45.98 | 22.26 | Evaluated |
| [RetinaNet R50](model_zoo/configs/bdd/bdd10k_ft_retinanet_r50.py) | 34.63 | 59.09 | 34.43 | Evaluated |
| [RTMDet-Tiny](model_zoo/configs/bdd/bdd10k_ft_rtmdet_tiny.py) | 26.91 | 47.62 | 25.44 | Evaluated |
| [Sparse R-CNN R50](model_zoo/configs/bdd/bdd10k_ft_sparse_rcnn_r50.py) | 8.96 | 19.11 | 7.08 | Evaluated |
| [SSD300](model_zoo/configs/bdd/bdd10k_ft_ssd300.py) | 6.77 | 13.82 | 6.06 | Evaluated |
| [TOOD R50](model_zoo/configs/bdd/bdd20k_ft_tood_r50.py) | 29.97 | 53.41 | 28.16 | Evaluated; class-name mapping |
| [TridentNet R50](model_zoo/configs/bdd/bdd20k_ft_tridentnet_r50.py) | 27.83 | 51.67 | 25.29 | Evaluated |
| [VarifocalNet R50](model_zoo/configs/bdd/bdd10k_ft_varifocalnet_r50.py) | 29.02 | 52.30 | 27.25 | Evaluated |
| [ViTDet-B](model_zoo/configs/bdd/bdd10k_ft_vitdet_b.py) | 31.90 | 56.01 | 30.88 | Evaluated |
| [YOLOF R50](model_zoo/configs/bdd/bdd10k_ft_yolof_r50.py) | 29.91 | 50.99 | 29.75 | Evaluated |
| [YOLOv3 Darknet-53 (320)](model_zoo/configs/bdd/bdd10k_ft_yolov3_320.py) | 8.53 | 20.93 | 6.16 | Evaluated |

## Other recorded revisions

Earlier runs and alternate recipes remain available separately; they do not add architecture slots.

| Run / config | AP | AP50 | AP75 | Status |
| --- | ---: | ---: | ---: | --- |
| [bdd10k_ft_efficientdet_d3](model_zoo/configs/bdd/bdd10k_ft_efficientdet_d3.py) | — | — | — | No detections |
| [bdd10k_ft_reppoints_r50](model_zoo/configs/bdd/bdd10k_ft_reppoints_r50.py) | — | — | — | No detections |
| [bdd10k_ft_reppoints_r50_v3](training/configs/bdd10k_ft_reppoints_r50_v3.py) | — | — | — | Training |

Native model test scales; COCO bbox AP with maxDets=100. Predictions use a 300-box cap where supported. See [evaluation details](docs/evaluation.md) and [known issues](docs/known-issues.md).
