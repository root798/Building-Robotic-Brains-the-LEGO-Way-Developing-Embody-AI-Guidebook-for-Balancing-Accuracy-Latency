# SeaDronesSee model zoo

[All domains](README.md) · [CSV](model_zoo/results.csv) · [Protocol](docs/evaluation.md) · [Training](docs/training-status.md)

Snapshot: 2026-09-20T17:04:05+00:00. AP values use a 0–100 scale.

| Train images | Validation images | Classes | Architecture slots |
| ---: | ---: | ---: | ---: |
| 8,930 | 1,547 | 5 | 32 |

Classes: `swimmer`, `boat`, `jetski`, `life_saving_appliances`, `buoy`.

## Models

Each row links to its recipe. Evaluated checkpoint filenames and SHA-256 identities are in the [catalog](model_zoo/catalog.json). Weight downloads are not hosted in this repository.

| Model | AP | AP50 | AP75 | Status |
| --- | ---: | ---: | ---: | --- |
| [ATSS R50](model_zoo/configs/seadronessee/seadronessee10k_ft_atss_r50.py) | 44.47 | 75.01 | 44.89 | Evaluated |
| [Cascade R-CNN R50](model_zoo/configs/seadronessee/seadronessee10k_cascade_rcnn_r50.py) | 38.83 | 60.90 | 42.44 | Evaluated |
| [CenterNet R18-DCN](model_zoo/configs/seadronessee/seadronessee10k_ft_centernet_r18_dcn.py) | 2.62 | 8.11 | 1.15 | Evaluated |
| [Conditional DETR R50](model_zoo/configs/seadronessee/seadronessee20k_ft_conditional_detr_r50.py) | 36.72 | 70.92 | 34.50 | Evaluated |
| [DDQ-DETR R50](model_zoo/configs/seadronessee/seadronessee10k_ft_ddq_detr4scale_r50.py) | 52.71 | 87.80 | 52.69 | Evaluated |
| [Deformable DETR R50](model_zoo/configs/seadronessee/seadronessee10k_ft_deformable_detr_r50.py) | 48.39 | 86.17 | 46.22 | Evaluated |
| [Deformable DETR Refine R50](model_zoo/configs/seadronessee/seadronessee10k_ft_deformable_detr_refine_r50.py) | 48.20 | 84.52 | 45.70 | Evaluated |
| [DETR R50](model_zoo/configs/seadronessee/seadronessee20k_ft_detr_r50.py) | 30.47 | 63.13 | 24.67 | Evaluated |
| [DiffusionDet R50](model_zoo/configs/seadronessee/seadronessee10k_ft_diffusiondet_r50_8xb2-50e_coco.py) | 52.01 | 86.48 | 52.60 | Evaluated |
| [DINO Swin-L](model_zoo/configs/seadronessee/seadronessee10k_ft_dino_swin_l.py) | 54.26 | 86.43 | 58.25 | Evaluated |
| [Dynamic R-CNN R50](model_zoo/configs/seadronessee/seadronessee10k_ft_dynamic_rcnn_r50.py) | 35.07 | 55.87 | 37.15 | Evaluated |
| [EfficientDet-D3](model_zoo/configs/seadronessee/seadronessee10k_ft_efficientdet_d3.py) | 27.63 | 52.43 | 25.40 | Evaluated |
| [FCOS R50](model_zoo/configs/seadronessee/seadronessee10k_ft_fcos_r50.py) | 43.60 | 75.15 | 43.40 | Evaluated |
| [Faster R-CNN R50-FPN](model_zoo/configs/seadronessee/seadronessee10k_ft_frcnn_r50_fpn.py) | 38.20 | 59.46 | 41.14 | Evaluated |
| [FreeAnchor R50 v2](model_zoo/configs/seadronessee/seadronessee10k_ft_freeanchor_r50_v2.py) | 30.19 | 55.57 | 29.00 | Evaluated |
| [Grid R-CNN R50](model_zoo/configs/seadronessee/seadronessee20k_ft_grid_rcnn_r50.py) | 39.31 | 60.43 | 42.27 | Evaluated |
| [Mask R-CNN Swin-T](model_zoo/configs/seadronessee/seadronessee10k_ft_mask_rcnn_swin_t.py) | 31.64 | 51.00 | 33.76 | Evaluated |
| [NAS-FCOS R50](model_zoo/configs/seadronessee/seadronessee20k_ft_nas_fcos_r50.py) | 44.36 | 74.57 | 45.13 | Evaluated |
| [PAA R50](model_zoo/configs/seadronessee/seadronessee20k_ft_paa_r50.py) | 50.89 | 85.68 | 51.96 | Evaluated |
| [RepPoints R50 v2](model_zoo/configs/seadronessee/seadronessee10k_ft_reppoints_r50_v2.py) | 28.76 | 53.74 | 27.84 | Evaluated |
| [RetinaNet EfficientNet-B3](model_zoo/configs/seadronessee/seadronessee10k_ft_retinanet_effb3.py) | 36.73 | 69.09 | 34.95 | Evaluated |
| [RetinaNet PVT-T](model_zoo/configs/seadronessee/seadronessee10k_ft_retinanet_pvtt.py) | 42.84 | 78.88 | 39.06 | Evaluated |
| [RetinaNet R50](model_zoo/configs/seadronessee/seadronessee10k_ft_retinanet_r50.py) | 45.07 | 79.90 | 44.43 | Evaluated |
| [RTMDet-Tiny](model_zoo/configs/seadronessee/seadronessee10k_ft_rtmdet_tiny.py) | 37.89 | 63.86 | 38.73 | Evaluated |
| [Sparse R-CNN R50](model_zoo/configs/seadronessee/seadronessee10k_sparse_rcnn_r50.py) | 30.26 | 50.31 | 32.10 | Evaluated |
| [SSD300](model_zoo/configs/seadronessee/seadronessee10k_ft_ssd300.py) | 25.23 | 55.52 | 19.99 | Evaluated |
| [TOOD R50](model_zoo/configs/seadronessee/seadronessee10k_ft_tood_r50.py) | 48.36 | 81.62 | 48.25 | Evaluated |
| [TridentNet R50](model_zoo/configs/seadronessee/seadronessee20k_ft_tridentnet_r50.py) | 37.40 | 61.50 | 39.99 | Evaluated |
| [VarifocalNet R50](model_zoo/configs/seadronessee/seadronessee10k_ft_varifocalnet_r50.py) | 46.96 | 77.56 | 47.54 | Evaluated |
| [ViTDet-B](model_zoo/configs/seadronessee/seadronessee10k_ft_vitdet_b.py) | 35.93 | 54.62 | 39.24 | Evaluated |
| [YOLOF R50](model_zoo/configs/seadronessee/seadronessee10k_ft_yolof_r50.py) | 23.70 | 43.75 | 22.29 | Evaluated |
| [YOLOv3 Darknet-53 (320)](model_zoo/configs/seadronessee/seadronessee10k_ft_yolov3_d53_320.py) | 18.66 | 46.20 | 12.19 | Evaluated |

## Other recorded revisions

Earlier runs and alternate recipes remain available separately; they do not add architecture slots.

| Run / config | AP | AP50 | AP75 | Status |
| --- | ---: | ---: | ---: | --- |
| [seadronessee10k_ft_faster_rcnn](model_zoo/configs/seadronessee/seadronessee10k_ft_faster_rcnn.py) | 38.17 | 59.52 | 40.96 | Evaluated |
| [seadronessee10k_ft_freeanchor_r50](model_zoo/configs/seadronessee/seadronessee10k_ft_freeanchor_r50.py) | 0.00 | 0.00 | 0.00 | Evaluated |
| [seadronessee10k_ft_reppoints_r50](model_zoo/configs/seadronessee/seadronessee10k_ft_reppoints_r50.py) | 0.00 | 0.00 | 0.00 | Evaluated |

Native model test scales; COCO bbox AP with maxDets=100. Predictions use a 300-box cap where supported. See [evaluation details](docs/evaluation.md) and [known issues](docs/known-issues.md).
