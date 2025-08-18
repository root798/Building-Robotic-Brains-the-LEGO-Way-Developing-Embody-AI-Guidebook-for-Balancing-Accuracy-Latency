#  WAID_FT: Fine-Tuned Object Detection Models for the Wildlife Aerial Images from Drone (WAID) Dataset

## 1. Dataset

| Split        | Images | Annotation source | Notes                                |
|--------------|-------:|-------------------|--------------------------------------|
| **Train**    | 11.1 k | WAID              | Wildlife aerial images               |
| **Val**      | 2.1 k  | WAID              | Used for all numbers in the table    |
| **Test**     | 1.2 k  | WAID              | Used for all numbers in the table    |

| **Attribute**        | **Details**                                           |
|----------------------|-------------------------------------------------------|
| **Total Images**     | 14,375 UAV aerial images                             |
| **Species (6)**      | Sheep, Cattle, Seals, Camels, Kiang, Zebras          |
| **Habitats**         | Deserts, Grasslands, Sandy beaches                   |
| **Conditions**       | Various weather, times of day, altitudes             |
| **Annotation**       | Box-level labeling with species classification       |
| **Focus**            | Small object detection in aerial imagery             |
| **Source**           | https://github.com/xiaohuicui/WAID                   |

The WAID dataset is a large-scale, multi-class dataset specifically designed for wildlife detection in UAV aerial imagery. The dataset is particularly challenging for small object detection tasks, as wildlife subjects often appear as small targets in aerial images captured at various altitudes and environmental conditions.

## 2. Models Zoo (↑ = best)
Below are fintuned model performance on WAID dataset:

| Model | bbox_mAP | bbox_mAP_50 | bbox_mAP_75 | bbox_mAP_s | bbox_mAP_m | bbox_mAP_l |
|:------|:--------:|:-----------:|:-----------:|:----------:|:----------:|:----------:|
| waid_ft_ddq_detr4scale_r50 | **0.616** | **0.972** | **0.697** | **0.474** | 0.630 | **0.714** |
| waid_ft_tood_r50 | 0.610 | 0.964 | 0.685 | 0.457 | **0.631** | 0.680 |
| waid_ft_varifocalnet_r50 | 0.602 | 0.959 | 0.669 | 0.443 | 0.627 | 0.676 |
| waid_ft_atss_r50 | 0.601 | 0.963 | 0.671 | 0.450 | 0.626 | 0.678 |
| waid_ft_diffusiondet_r50_8xb2-50e_coco | 0.601 | 0.962 | 0.675 | 0.460 | 0.624 | 0.694 |
| waid_ft_paa_r50 | 0.598 | 0.958 | 0.667 | 0.434 | 0.624 | 0.679 |
| waid_ft_grid_rcnn_r50 | 0.596 | 0.950 | 0.676 | 0.452 | 0.616 | 0.698 |
| waid_ft_dino_swin_l | 0.593 | 0.964 | 0.659 | 0.461 | 0.609 | 0.679 |
| waid_cascade_rcnn_r50 | 0.590 | 0.949 | 0.664 | 0.441 | 0.609 | 0.683 |
| waid_ft_retinanet_r50 | 0.586 | 0.946 | 0.650 | 0.423 | 0.620 | 0.685 |
| waid_ft_nas_fcos_r50 | 0.586 | 0.957 | 0.646 | 0.441 | 0.610 | 0.682 |
| waid_ft_frcnn_r50_fpn | 0.584 | 0.956 | 0.648 | 0.437 | 0.606 | 0.658 |
| waid_ft_rtmdet_tiny | 0.583 | 0.938 | 0.645 | 0.403 | 0.611 | 0.680 |
| waid_ft_deformable_detr_r50 | 0.581 | 0.966 | 0.628 | 0.439 | 0.601 | 0.670 |
| waid_ft_tridentnet_r50 | 0.577 | 0.948 | 0.632 | 0.412 | 0.611 | 0.669 |
| waid_ft_faster_rcnn | 0.574 | 0.951 | 0.628 | 0.423 | 0.600 | 0.635 |
| waid_ft_fcos_r50 | 0.574 | 0.952 | 0.627 | 0.426 | 0.600 | 0.685 |
| waid_ft_retinanet_effb3 | 0.565 | 0.928 | 0.622 | 0.368 | 0.605 | 0.672 |
| waid_ft_retinanet_pvtt | 0.556 | 0.940 | 0.596 | 0.399 | 0.592 | 0.644 |
| waid_ft_mask_rcnn_swin_t | 0.547 | 0.940 | 0.588 | 0.407 | 0.576 | 0.606 |
| waid_ft_yolof_r50 | 0.537 | 0.905 | 0.577 | 0.334 | 0.611 | 0.679 |
| waid_ft_conditional_detr_r50 | 0.494 | 0.894 | 0.490 | 0.304 | 0.560 | 0.647 |
| waid_ft_detr_r50 | 0.455 | 0.850 | 0.440 | 0.239 | 0.531 | 0.620 |
| waid_ft_dynamic_rcnn_r50 | 0.402 | 0.694 | 0.425 | 0.275 | 0.447 | 0.355 |
| waid_sparse_rcnn_r50 | 0.257 | 0.497 | 0.245 | 0.125 | 0.310 | 0.276 |
| waid_ft_yolov3_d53_320 | 0.244 | 0.603 | 0.142 | 0.107 | 0.297 | 0.401 |
| waid_ft_efficientdet_d3 | 0.173 | 0.303 | 0.184 | 0.094 | 0.218 | 0.196 |
| waid_ft_freeanchor_r50_smallobj_fp32   |      0.135 |         0.293 |         0.101 |        0.116 |        0.152 |        0.183 |
| waid_ft_centernet_r18_dcn | 0.111 | 0.284 | 0.066 | 0.044 | 0.147 | 0.154 |
| waid_ft_reppoints_r50 | 0.014 | 0.041 | 0.006 | 0.006 | 0.026 | 0.000 |


**Note**: Bold values indicate the best performance in each metric category among WAID models.

## 3. Citation
For datasets used in the fintuning and scenario generation, these are the corresponding resources:

```bibtex
@article{mou2023waid,
  title={WAID: A Large-Scale Dataset for Wildlife Detection with Drones},
  author={Mou, Chao and Liu, Tengfei and Zhu, Chengcheng and Cui, Xiaohui},
  journal={Applied Sciences},
  volume={13},
  number={18},
  pages={10397},
  year={2023},
  publisher={MDPI},
  doi={10.3390/app131810397}
}
```
