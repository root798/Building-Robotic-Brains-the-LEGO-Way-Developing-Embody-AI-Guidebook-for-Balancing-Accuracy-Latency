# SeaDronesSee Dataset
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
| seadronessee10k_ft_ddq_detr4scale_r50 | 0.527 | 0.878 | 0.527 | 0.477 | 0.533 | 0.706 |
| seadronessee10k_ft_varifocalnet_r50 | 0.453 | 0.769 | 0.460 | 0.341 | 0.494 | 0.674 |
| seadronessee10k_ft_retinanet_r50 | 0.451 | 0.799 | 0.444 | 0.339 | 0.475 | 0.657 |
| seadronessee10k_ft_atss_r50 | 0.444 | 0.751 | 0.451 | 0.360 | 0.490 | 0.644 |
| seadronessee10k_cascade_rcnn_r50 | 0.400 | 0.626 | 0.435 | 0.249 | 0.483 | 0.669 |
| seadronessee20k_ft_grid_rcnn_r50 | 0.393 | 0.605 | 0.428 | 0.213 | 0.470 | 0.675 |
| seadronessee10k_ft_frcnn_r50_fpn | 0.378 | 0.598 | 0.409 | 0.159 | 0.456 | 0.651 |
| seadronessee20k_ft_tridentnet_r50 | 0.369 | 0.596 | 0.399 | 0.180 | 0.427 | 0.643 |
| seadronessee10k_ft_retinanet_effb3 | 0.369 | 0.691 | 0.360 | 0.184 | 0.388 | 0.661 |
| seadronessee20k_ft_conditional_detr_r50 | 0.367 | 0.709 | 0.345 | 0.255 | 0.390 | 0.613 |
| seadronessee10k_ft_dynamic_rcnn_r50 | 0.348 | 0.552 | 0.370 | 0.107 | 0.418 | 0.612 |
| seadronessee10k_ft_retinanet_pvtt | 0.340 | 0.647 | 0.320 | 0.181 | 0.363 | 0.628 |
| seadronessee20k_ft_detr_r50 | 0.305 | 0.632 | 0.247 | 0.163 | 0.305 | 0.541 |
| seadronessee10k_sparse_rcnn_r50 | 0.303 | 0.503 | 0.321 | 0.278 | 0.355 | 0.460 |


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
