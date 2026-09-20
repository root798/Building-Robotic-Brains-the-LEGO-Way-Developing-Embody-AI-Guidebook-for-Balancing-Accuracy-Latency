# SeaDronesSee Dataset
> Historical model table: these values are not results of the current completion
> campaign. See [current training status](docs/training-status.md) for the dated
> run inventory and [reproduction](docs/reproduction.md) for published recipes.

## 1. Dataset
| Split        | Images | Annotation source | Notes                                |
|--------------|-------:|-------------------|--------------------------------------|
| **Train**    | 8.9 k  | SeaDronesSee v2   | Maritime SAR aerial images           |
| **Val**      | 1.5 k  | SeaDronesSee v2   | Used for all numbers in the table    |
| **Test**     | 3.8 k  | SeaDronesSee v2   | Used for all numbers in the table    |

| **Attribute**        | **Details**                                           |
|----------------------|-------------------------------------------------------|
| **Total Images**     | 14,227 RGB aerial images                             |
| **Classes (5)**      | Swimmer, Boat, Jetski, Buoy, Life saving appliance  |
| **Environment**      | Open water, maritime scenes                          |
| **Conditions**       | Various altitudes (5-260m), viewing angles (0-90°)  |
| **Resolutions**      | 3840x2160, 5456x3632, 1229x934                      |
| **Annotation**       | Axis-aligned bounding boxes + ignore regions         |
| **Focus**            | Search and Rescue (SAR) object detection             |
| **Source**           | https://macvi.org/workshop/macvi23/                  |

The SeaDronesSee dataset is a maritime benchmark specifically designed for Search and Rescue (SAR) missions, focusing on detecting humans and objects in open water from UAV perspectives. The dataset presents unique challenges for object detection in maritime environments, with the best performing models currently achieving only 36% mAP compared to 60%+ on COCO, highlighting the difficulty of this domain. The dataset includes comprehensive metadata for altitude, viewing angles, and other flight parameters for most frames.




## 2. Models Zoo (↑ = best)
| Model | bbox_mAP | bbox_mAP_50 | bbox_mAP_75 | bbox_mAP_s | bbox_mAP_m | bbox_mAP_l |
|:------|:--------:|:-----------:|:-----------:|:----------:|:----------:|:----------:|
| seadronessee_ft_ddq_detr4scale_r50 | 0.527 | 0.878 | 0.527 | 0.477 | 0.533 | 0.706 |
| seadronessee_ft_tood_r50 | 0.518 | 0.867 | 0.528 | 0.388 | 0.536 | 0.678 |
| seadronessee_ft_paa_r50 | 0.509 | 0.858 | 0.519 | 0.445 | 0.511 | 0.689 |
| seadronessee_ft_diffusiondet_r50 | 0.492 | 0.818 | 0.506 | 0.377 | 0.512 | 0.660 |
| seadronessee_ft_dino_swin_l | 0.486 | 0.851 | 0.485 | 0.416 | 0.508 | 0.666 |
| seadronessee_ft_varifocalnet_r50 | 0.474 | 0.782 | 0.480 | 0.377 | 0.509 | 0.674 |
| seadronessee_ft_rtmdet_tiny | 0.467 | 0.797 | 0.484 | 0.322 | 0.489 | 0.612 |
| seadronessee_ft_deformable_detr_r50 | 0.465 | 0.821 | 0.471 | 0.351 | 0.481 | 0.603 |
| seadronessee_ft_retinanet_r50 | 0.451 | 0.799 | 0.444 | 0.339 | 0.475 | 0.657 |
| seadronessee_ft_atss_r50 | 0.444 | 0.751 | 0.451 | 0.360 | 0.490 | 0.644 |
| seadronessee_ft_nas_fcos_r50 | 0.444 | 0.747 | 0.452 | 0.295 | 0.482 | 0.666 |
| seadronessee_ft_fcos_r50 | 0.430 | 0.762 | 0.439 | 0.320 | 0.450 | 0.617 |
| seadronessee_cascade_rcnn_r50 | 0.400 | 0.626 | 0.435 | 0.249 | 0.483 | 0.669 |
| seadronessee_ft_grid_rcnn_r50 | 0.393 | 0.605 | 0.428 | 0.213 | 0.470 | 0.675 |
| seadronessee_ft_frcnn_r50_fpn | 0.378 | 0.598 | 0.409 | 0.159 | 0.456 | 0.651 |
| seadronessee_ft_faster_rcnn | 0.375 | 0.621 | 0.410 | 0.276 | 0.392 | 0.415 |
| seadronessee_ft_tridentnet_r50 | 0.369 | 0.596 | 0.399 | 0.180 | 0.427 | 0.643 |
| seadronessee_ft_retinanet_effb3 | 0.369 | 0.691 | 0.360 | 0.184 | 0.388 | 0.661 |
| seadronessee_ft_conditional_detr_r50 | 0.367 | 0.709 | 0.345 | 0.255 | 0.390 | 0.613 |
| seadronessee_ft_dynamic_rcnn_r50 | 0.348 | 0.552 | 0.370 | 0.107 | 0.418 | 0.612 |
| seadronessee_ft_retinanet_pvtt | 0.340 | 0.647 | 0.320 | 0.181 | 0.363 | 0.628 |
| seadronessee_ft_mask_rcnn_swin_t | 0.311 | 0.495 | 0.334 | 0.090 | 0.334 | 0.599 |
| seadronessee_ft_detr_r50 | 0.305 | 0.632 | 0.247 | 0.163 | 0.305 | 0.541 |
| seadronessee_sparse_rcnn_r50 | 0.303 | 0.503 | 0.321 | 0.278 | 0.355 | 0.460 |
| seadronessee_ft_efficientdet_d3 | 0.277 | 0.522 | 0.256 | 0.155 | 0.304 | 0.489 |
| seadronessee_ft_yolof_r50 | 0.233 | 0.428 | 0.220 | 0.065 | 0.270 | 0.465 |
| seadronessee_ft_yolov3_d53_320 | 0.185 | 0.456 | 0.121 | 0.061 | 0.221 | 0.345 |
| seadronessee_ft_freeanchor_r50_smallobj_fp32 | 0.088 | 0.191 | 0.066 | 0.076 | 0.099 | 0.119 |
| seadronessee_ft_centernet_r18_dcn | 0.030 | 0.095 | 0.013 | 0.004 | 0.006 | 0.072 |
| seadronessee_ft_reppoints_r50 | 0.009 | 0.027 | 0.004 | 0.004 | 0.017 | 0.000 |



## 3. Citation
For dataset used in the fintuning and scenario generation, below is the corresponding resource:

```bibtex
@inproceedings{varga2022seadronessee,
title={Seadronessee: A maritime benchmark for detecting humans in open water},
author={Varga, Leon Amadeus and Kiefer, Benjamin and Messmer, Martin and Zell, Andreas},
booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
pages={2260--2270},
year={2022} }
```
