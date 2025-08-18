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
