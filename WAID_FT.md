# WAID model zoo

[All domains](README.md) · [CSV](model_zoo/results.csv) · [Protocol](docs/evaluation.md) · [Training](docs/training-status.md)

Snapshot: 2026-09-20T17:04:05+00:00. AP values use a 0–100 scale.

| Train images | Validation images | Classes | Architecture slots |
| ---: | ---: | ---: | ---: |
| 10,056 | 2,873 | 6 | 32 |

Classes: `sheep`, `cattle`, `seal`, `camelus`, `kiang`, `zebra`.

## Models

Each row links to its recipe. Evaluated checkpoint filenames and SHA-256 identities are in the [catalog](model_zoo/catalog.json). Weight downloads are not hosted in this repository.

| Model | AP | AP50 | AP75 | Status |
| --- | ---: | ---: | ---: | --- |
| [ATSS R50](model_zoo/configs/waid/waid_ft_atss_r50.py) | 60.01 | 96.26 | 67.16 | Evaluated |
| [Cascade R-CNN R50](model_zoo/configs/waid/waid10k_cascade_rcnn_r50.py) | 59.23 | 95.10 | 66.43 | Evaluated |
| [CenterNet R18-DCN](model_zoo/configs/waid/waid10k_ft_centernet_r18_dcn.py) | 11.08 | 28.35 | 6.60 | Evaluated |
| [Conditional DETR R50](model_zoo/configs/waid/waid20k_ft_conditional_detr_r50.py) | 49.40 | 89.37 | 49.04 | Evaluated |
| [DDQ-DETR R50](model_zoo/configs/waid/waid10k_ft_ddq_detr4scale_r50.py) | 61.82 | 97.16 | 69.94 | Evaluated |
| [Deformable DETR R50](model_zoo/configs/waid/waid10k_ft_deformable_detr_r50.py) | 58.10 | 96.57 | 62.48 | Evaluated |
| [Deformable DETR Refine R50](model_zoo/configs/waid/waid10k_ft_deformable_detr_refine_r50.py) | 57.68 | 95.26 | 62.97 | Evaluated |
| [DETR R50](model_zoo/configs/waid/waid20k_ft_detr_r50.py) | 45.46 | 85.00 | 44.04 | Evaluated |
| [DiffusionDet R50](model_zoo/configs/waid/waid10k_ft_diffusiondet_r50_8xb2-50e_coco.py) | 59.91 | 96.05 | 66.89 | Evaluated |
| [DINO Swin-L](model_zoo/configs/waid/waid10k_ft_dino_swin_l.py) | 59.01 | 96.23 | 65.69 | Evaluated |
| [Dynamic R-CNN R50](model_zoo/configs/waid/waid10k_ft_dynamic_rcnn_r50.py) | 40.56 | 70.18 | 42.67 | Evaluated |
| [EfficientDet-D3](model_zoo/configs/waid/waid10k_ft_efficientdet_d3.py) | 15.68 | 27.98 | 16.17 | Evaluated |
| [FCOS R50](model_zoo/configs/waid/waid10k_ft_fcos_r50.py) | 57.40 | 95.18 | 62.08 | Evaluated |
| [Faster R-CNN R50-FPN](model_zoo/configs/waid/waid10k_ft_frcnn_r50_fpn.py) | 58.52 | 95.87 | 64.83 | Evaluated |
| [FreeAnchor R50 v2](model_zoo/configs/waid/waid10k_ft_freeanchor_r50_v2.py) | 32.18 | 57.13 | 32.87 | Evaluated |
| [Grid R-CNN R50](model_zoo/configs/waid/waid20k_ft_grid_rcnn_r50.py) | 59.62 | 94.90 | 67.65 | Evaluated |
| [Mask R-CNN Swin-T](model_zoo/configs/waid/waid10k_ft_mask_rcnn_swin_t.py) | 54.91 | 94.43 | 58.90 | Evaluated |
| [NAS-FCOS R50](model_zoo/configs/waid/waid20k_ft_nas_fcos_r50.py) | 58.58 | 95.67 | 64.55 | Evaluated |
| [PAA R50](model_zoo/configs/waid/waid20k_ft_paa_r50.py) | 59.79 | 95.82 | 66.71 | Evaluated |
| [RepPoints R50 v2](model_zoo/configs/waid/waid10k_ft_reppoints_r50_v2.py) | 30.00 | 60.19 | 26.28 | Evaluated |
| [RetinaNet EfficientNet-B3](model_zoo/configs/waid/waid10k_ft_retinanet_effb3.py) | 56.38 | 92.58 | 62.14 | Evaluated |
| [RetinaNet PVT-T](model_zoo/configs/waid/waid10k_ft_retinanet_pvtt.py) | 55.54 | 93.93 | 59.28 | Evaluated |
| [RetinaNet R50](model_zoo/configs/waid/waid10k_ft_retinanet_r50.py) | 58.61 | 94.66 | 65.05 | Evaluated |
| [RTMDet-Tiny](model_zoo/configs/waid/waid10k_ft_rtmdet_tiny.py) | 58.34 | 93.75 | 64.47 | Evaluated |
| [Sparse R-CNN R50](model_zoo/configs/waid/waid10k_sparse_rcnn_r50.py) | 25.71 | 49.77 | 24.43 | Evaluated |
| [SSD300](model_zoo/configs/waid/waid10k_ft_ssd300.py) | 4.92 | 23.06 | 0.52 | Evaluated |
| [TOOD R50](model_zoo/configs/waid/waid20k_ft_tood_r50.py) | 61.10 | 96.42 | 68.62 | Evaluated |
| [TridentNet R50](model_zoo/configs/waid/waid20k_ft_tridentnet_r50.py) | 57.78 | 95.07 | 63.15 | Evaluated |
| [VarifocalNet R50](model_zoo/configs/waid/waid10k_ft_varifocalnet_r50.py) | 60.23 | 95.87 | 66.95 | Evaluated |
| [ViTDet-B](training/configs/waid10k_ft_vitdet_b.py) | — | — | — | Training |
| [YOLOF R50](model_zoo/configs/waid/waid10k_ft_yolof_r50.py) | 53.69 | 90.49 | 57.57 | Evaluated |
| [YOLOv3 Darknet-53 (320)](model_zoo/configs/waid/waid10k_ft_yolov3_d53_320.py) | 24.68 | 61.03 | 14.41 | Evaluated |

## Other recorded revisions

Earlier runs and alternate recipes remain available separately; they do not add architecture slots.

| Run / config | AP | AP50 | AP75 | Status |
| --- | ---: | ---: | ---: | --- |
| [waid10k_ft_faster_rcnn](model_zoo/configs/waid/waid10k_ft_faster_rcnn.py) | 57.53 | 95.36 | 62.77 | Evaluated |
| [waid10k_ft_freeanchor_r50](model_zoo/configs/waid/waid10k_ft_freeanchor_r50.py) | 0.00 | 0.00 | 0.00 | Evaluated |
| [waid10k_ft_freeanchor_r50_smallobj_fp32](model_zoo/configs/waid/waid10k_ft_freeanchor_r50_smallobj_fp32.py) | 14.31 | 31.18 | 10.75 | Evaluated |
| [waid10k_ft_reppoints_r50](model_zoo/configs/waid/waid10k_ft_reppoints_r50.py) | 1.49 | 4.31 | 0.65 | Evaluated |

Native model test scales; COCO bbox AP with maxDets=100. Predictions use a 300-box cap where supported. See [evaluation details](docs/evaluation.md) and [known issues](docs/known-issues.md).
