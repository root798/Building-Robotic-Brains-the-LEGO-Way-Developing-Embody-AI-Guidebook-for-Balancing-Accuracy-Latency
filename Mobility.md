# Mobility / CMA model zoo

[All domains](README.md) · [CSV](model_zoo/results.csv) · [Protocol](docs/evaluation.md) · [Training](docs/training-status.md)

Snapshot: 2026-09-20T17:04:05+00:00. AP values use a 0–100 scale.

| Train images | Validation images | Classes | Architecture slots |
| ---: | ---: | ---: | ---: |
| 8,456 | 784 | 5 | 32 |

Classes: `person`, `wheelchair`, `rollator`, `crutch`, `cane`.

## Models

Each row links to its recipe. Evaluated checkpoint filenames and SHA-256 identities are in the [catalog](model_zoo/catalog.json). Weight downloads are not hosted in this repository.

| Model | AP | AP50 | AP75 | Status |
| --- | ---: | ---: | ---: | --- |
| [ATSS R50](model_zoo/configs/mobility/mobility10k_ft_atss_r50.py) | 68.30 | 87.31 | 77.29 | Evaluated |
| [Cascade R-CNN R50](model_zoo/configs/mobility/mobility10k_cascade_rcnn_r50.py) | 65.26 | 86.73 | 75.28 | Evaluated |
| [CenterNet R18-DCN](model_zoo/configs/mobility/mobility10k_ft_centernet_r18_dcn.py) | 14.91 | 33.14 | 11.20 | Evaluated |
| [Conditional DETR R50](model_zoo/configs/mobility/mobility20k_ft_conditional_detr_r50i.py) | 48.80 | 80.59 | 55.63 | Evaluated |
| [DDQ-DETR R50](model_zoo/configs/mobility/mobility10k_ft_ddq_detr4scale_r50.py) | 70.50 | 87.29 | 79.07 | Evaluated |
| [Deformable DETR R50](model_zoo/configs/mobility/mobility10k_ft_deformable_detr_r50.py) | 65.99 | 86.17 | 76.15 | Evaluated |
| [Deformable DETR Refine R50](model_zoo/configs/mobility/deformable-detr-refine_r50_2xb2-25e_mobility100k.py) | 66.31 | 86.14 | 75.56 | Evaluated |
| [DETR R50](model_zoo/configs/mobility/mobility20k_ft_detr_r50.py) | 44.69 | 67.70 | 51.97 | Evaluated |
| [DiffusionDet R50](model_zoo/configs/mobility/mobility10k_ft_diffusiondet_r50.py) | 67.11 | 87.92 | 75.38 | Evaluated |
| [DINO Swin-L](model_zoo/configs/mobility/mobility10k_ft_dino_swin_ln.py) | 72.52 | 90.32 | 80.17 | Evaluated |
| [Dynamic R-CNN R50](model_zoo/configs/mobility/mobility10k_ft_dynamic_rcnn_r50.py) | 60.67 | 81.97 | 70.27 | Evaluated |
| [EfficientDet-D3](model_zoo/configs/mobility/mobility10k_ft_efficientdet_d3.py) | 55.22 | 74.82 | 62.46 | Evaluated |
| [FCOS R50](model_zoo/configs/mobility/mobility10k_ft_fcos_r50.py) | 64.53 | 85.79 | 74.30 | Evaluated |
| [Faster R-CNN R50-FPN](model_zoo/configs/mobility/mobility10k_ft_faster_rcnn.py) | 59.20 | 84.88 | 69.33 | Evaluated |
| [FreeAnchor R50](model_zoo/configs/mobility/mobility10k_ft_freeanchor_r50.py) | 33.00 | 53.43 | 35.78 | Evaluated |
| [Grid R-CNN R50](model_zoo/configs/mobility/mobility20k_ft_grid_rcnn_r50.py) | 65.16 | 83.75 | 74.57 | Evaluated |
| [Mask R-CNN Swin-T](model_zoo/configs/mobility/mobility10k_ft_mask_rcnn_swin_t.py) | 58.68 | 83.69 | 68.79 | Evaluated |
| [NAS-FCOS R50](model_zoo/configs/mobility/mobility20k_ft_nas_fcos_r50.py) | 65.37 | 85.18 | 73.18 | Evaluated |
| [PAA R50](model_zoo/configs/mobility/mobility20k_ft_paa_r50.py) | 69.23 | 88.10 | 78.79 | Evaluated |
| [RepPoints R50](model_zoo/configs/mobility/mobility10k_ft_reppoints_r50.py) | 33.58 | 54.41 | 36.64 | Evaluated |
| [RetinaNet EfficientNet-B3](model_zoo/configs/mobility/mobility10k_ft_retinanet_effb3.py) | 64.43 | 84.46 | 72.97 | Evaluated |
| [RetinaNet PVT-T](model_zoo/configs/mobility/mobility10k_ft_retinanet_pvtt.py) | 60.29 | 81.99 | 69.06 | Evaluated |
| [RetinaNet R50](model_zoo/configs/mobility/mobility10k_ft_retinanet_r50.py) | 66.14 | 86.49 | 75.07 | Evaluated |
| [RTMDet-Tiny](model_zoo/configs/mobility/mobility10k_ft_rtmdet_tiny.py) | 59.96 | 81.26 | 68.71 | Evaluated |
| [Sparse R-CNN R50](model_zoo/configs/mobility/mobility10k_sparse_rcnn_r50.py) | 53.18 | 75.55 | 60.16 | Evaluated |
| [SSD300](model_zoo/configs/mobility/mobility10k_ft_ssd300.py) | 40.91 | 65.67 | 46.11 | Evaluated |
| [TOOD R50](model_zoo/configs/mobility/mobility20k_ft_tood_r50.py) | 70.97 | 88.75 | 79.17 | Evaluated |
| [TridentNet R50](model_zoo/configs/mobility/mobility20k_ft_tridentnet_r50.py) | 62.21 | 84.99 | 71.84 | Evaluated |
| [VarifocalNet R50](model_zoo/configs/mobility/mobility10k_ft_varifocalnet_r50.py) | 69.34 | 87.27 | 78.02 | Evaluated |
| [ViTDet-B](model_zoo/configs/mobility/mobility10k_vitdet_b.py) | 72.82 | 91.43 | 81.88 | Evaluated |
| [YOLOF R50](model_zoo/configs/mobility/mobility10k_ft_yolof_r50.py) | 62.68 | 83.80 | 72.29 | Evaluated |
| [YOLOv3 Darknet-53 (320)](model_zoo/configs/mobility/mobility10k_ft_yolov3_d53_320.py) | 26.45 | 51.29 | 24.50 | Evaluated |

Native model test scales; COCO bbox AP with maxDets=100. Predictions use a 300-box cap where supported. See [evaluation details](docs/evaluation.md) and [known issues](docs/known-issues.md).
