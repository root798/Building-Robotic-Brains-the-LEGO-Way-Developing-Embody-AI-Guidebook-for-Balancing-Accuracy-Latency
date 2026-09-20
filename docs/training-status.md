# Fine-tuning progress

Snapshot: 2026-09-20T17:06:30+00:00.

The completion campaign contains 20 jobs: 18 completed with finite logged losses, one completed with a numerical issue, and one still training. BDD RepPoints v3 is a separate replacement candidate.

[Full model zoo](../README.md) · [Machine-readable status](../training/status.json)

| Domain | Recipe | Steps / schedule | Status |
| --- | --- | ---: | --- |
| mobility | [mobility10k_ft_fcos_r50](../training/configs/mobility10k_ft_fcos_r50.py) | 11,532 / 11,532 | Completed |
| mobility | [mobility10k_ft_atss_r50](../training/configs/mobility10k_ft_atss_r50.py) | 11,532 / 11,532 | Completed |
| mobility | [mobility10k_ft_dynamic_rcnn_r50](../training/configs/mobility10k_ft_dynamic_rcnn_r50.py) | 11,532 / 11,532 | Completed |
| mobility | [mobility10k_ft_rtmdet_tiny](../training/configs/mobility10k_ft_rtmdet_tiny.py) | 23,064 / 23,064 | Completed |
| mobility | [mobility10k_ft_yolof_r50](../training/configs/mobility10k_ft_yolof_r50.py) | 23,064 / 23,064 | Completed |
| mobility | [mobility10k_ft_ssd300](../training/configs/mobility10k_ft_ssd300.py) | 23,064 / 23,064 | Completed |
| mobility | [mobility10k_ft_deformable_detr_r50](../training/configs/mobility10k_ft_deformable_detr_r50.py) | 92,256 / 92,256 | Completed |
| mobility | [mobility10k_ft_mask_rcnn_swin_t](../training/configs/mobility10k_ft_mask_rcnn_swin_t.py) | 30,752 / 30,752 | Completed |
| mobility | [mobility10k_ft_diffusiondet_r50](../training/configs/mobility10k_ft_diffusiondet_r50.py) | 50,000 / 50,000 | Completed |
| seadronessee | [seadronessee10k_ft_ssd300](../training/configs/seadronessee10k_ft_ssd300.py) | 26,808 / 26,808 | Completed |
| seadronessee | [seadronessee10k_ft_deformable_detr_refine_r50](../training/configs/seadronessee10k_ft_deformable_detr_refine_r50.py) | 107,160 / 107,160 | Completed |
| seadronessee | [seadronessee10k_ft_vitdet_b](../training/configs/seadronessee10k_ft_vitdet_b.py) | 214,320 / 214,320 | Completed |
| waid | [waid10k_ft_deformable_detr_refine_r50](../training/configs/waid10k_ft_deformable_detr_refine_r50.py) | 120,672 / 120,672 | Completed |
| waid | [waid10k_ft_vitdet_b](../training/configs/waid10k_ft_vitdet_b.py) | 131,178 / 241,344 | Training |
| bdd | [bdd10k_ft_reppoints_r50_v2](../training/configs/bdd10k_ft_reppoints_r50_v2.py) | 18,000 / 18,000 | Numerical issue |
| bdd | [bdd10k_ft_efficientdet_d3_v2](../training/configs/bdd10k_ft_efficientdet_d3_v2.py) | 72,000 / 72,000 | Completed |
| seadronessee | [seadronessee10k_ft_reppoints_r50_v2](../training/configs/seadronessee10k_ft_reppoints_r50_v2.py) | 13,404 / 13,404 | Completed |
| seadronessee | [seadronessee10k_ft_freeanchor_r50_v2](../training/configs/seadronessee10k_ft_freeanchor_r50_v2.py) | 13,404 / 13,404 | Completed |
| waid | [waid10k_ft_reppoints_r50_v2](../training/configs/waid10k_ft_reppoints_r50_v2.py) | 15,084 / 15,084 | Completed |
| waid | [waid10k_ft_freeanchor_r50_v2](../training/configs/waid10k_ft_freeanchor_r50_v2.py) | 15,084 / 15,084 | Completed |
| bdd | [bdd10k_ft_reppoints_r50_v3](../training/configs/bdd10k_ft_reppoints_r50_v3.py) | 14,400 / 18,000 | Training |

Completed schedules are checked against final checkpoint-save and validation records. Loss status comes from periodic training logs. The v3 repair additionally checks every training batch.

Accuracy results are in the domain model tables. See [numerical stability](numerical-stability.md) for the RepPoints repair and [dataset details](datasets.md) for the project splits.
