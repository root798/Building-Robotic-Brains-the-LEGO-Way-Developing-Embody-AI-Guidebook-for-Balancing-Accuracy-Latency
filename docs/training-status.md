# Four-domain fine-tuning status

Snapshot: **2026-09-20T16:32:04Z**. This is a dated, manually audited snapshot, not a live dashboard. Machine-readable source: [training/status.json](../training/status.json).

## What is counted

The current queue contains **20 formal fill-in/replacement jobs**: 9 Mobility, 5 SeaDronesSee, 4 WAID, and 2 BDD jobs. These are additions or replacements within a larger detector collection; they are not the entire collection, and this snapshot does not certify a completed 32 × 4 grid.

At the snapshot:

- **18 jobs completed with no nonfinite loss found in their selected training logs.**
- **1 completed job has a known numerical defect:** BDD RepPoints v2 has 23 NaN reporting windows and must not be described as clean training.
- **1 formal job is still training:** WAID ViTDet-B.
- A separate BDD RepPoints v3 repair candidate is running. It is a candidate replacement for v2, **not a 21st formal job**.

## Job inventory

“Steps” means training-loop iterations, not guaranteed successful optimizer updates. For completed jobs, final-schedule checkpoint-save and validation records establish schedule completion; periodic training logs can end a few steps before the final iteration. For running jobs, the latest logged step is a lower bound on live progress.

| Domain | Model / config | Completed steps / schedule | Log-level status |
|---|---|---:|---|
| Mobility / CMA | [FCOS R50](../training/configs/mobility10k_ft_fcos_r50.py) | 11,532 / 11,532 | Completed; no logged nonfinite loss |
| Mobility / CMA | [ATSS R50](../training/configs/mobility10k_ft_atss_r50.py) | 11,532 / 11,532 | Completed; no logged nonfinite loss |
| Mobility / CMA | [Dynamic R-CNN R50](../training/configs/mobility10k_ft_dynamic_rcnn_r50.py) | 11,532 / 11,532 | Completed; no logged nonfinite loss |
| Mobility / CMA | [RTMDet-Tiny](../training/configs/mobility10k_ft_rtmdet_tiny.py) | 23,064 / 23,064 | Completed; no logged nonfinite loss |
| Mobility / CMA | [YOLOF R50](../training/configs/mobility10k_ft_yolof_r50.py) | 23,064 / 23,064 | Completed; no logged nonfinite loss |
| Mobility / CMA | [SSD300](../training/configs/mobility10k_ft_ssd300.py) | 23,064 / 23,064 | Completed; no logged nonfinite loss |
| Mobility / CMA | [Deformable DETR R50](../training/configs/mobility10k_ft_deformable_detr_r50.py) | 92,256 / 92,256 | Completed; no logged nonfinite loss |
| Mobility / CMA | [Mask R-CNN Swin-T](../training/configs/mobility10k_ft_mask_rcnn_swin_t.py) | 30,752 / 30,752 | Completed; no logged nonfinite loss |
| Mobility / CMA | [DiffusionDet R50](../training/configs/mobility10k_ft_diffusiondet_r50.py) | 50,000 / 50,000 | Completed; no logged nonfinite loss |
| SeaDronesSee | [SSD300](../training/configs/seadronessee10k_ft_ssd300.py) | 26,808 / 26,808 | Completed; no logged nonfinite loss |
| SeaDronesSee | [Deformable DETR Refine R50](../training/configs/seadronessee10k_ft_deformable_detr_refine_r50.py) | 107,160 / 107,160 | Completed; no logged nonfinite loss |
| SeaDronesSee | [ViTDet-B](../training/configs/seadronessee10k_ft_vitdet_b.py) | 214,320 / 214,320 | Completed; no logged nonfinite loss |
| WAID | [Deformable DETR Refine R50](../training/configs/waid10k_ft_deformable_detr_refine_r50.py) | 120,672 / 120,672 | Completed; no logged nonfinite loss |
| WAID | [ViTDet-B](../training/configs/waid10k_ft_vitdet_b.py) | 124,172 / 241,344 | Training; no logged nonfinite loss so far |
| BDD project split | [RepPoints R50 v2](../training/configs/bdd10k_ft_reppoints_r50_v2.py) | 18,000 / 18,000 | Completed; **23 NaN windows** |
| BDD project split | [EfficientDet-D3 v2](../training/configs/bdd10k_ft_efficientdet_d3_v2.py) | 72,000 / 72,000 | Completed; no logged nonfinite loss |
| SeaDronesSee | [RepPoints R50 v2](../training/configs/seadronessee10k_ft_reppoints_r50_v2.py) | 13,404 / 13,404 | Completed; no logged nonfinite loss |
| SeaDronesSee | [FreeAnchor R50 v2](../training/configs/seadronessee10k_ft_freeanchor_r50_v2.py) | 13,404 / 13,404 | Completed; no logged nonfinite loss |
| WAID | [RepPoints R50 v2](../training/configs/waid10k_ft_reppoints_r50_v2.py) | 15,084 / 15,084 | Completed; no logged nonfinite loss |
| WAID | [FreeAnchor R50 v2](../training/configs/waid10k_ft_freeanchor_r50_v2.py) | 15,084 / 15,084 | Completed; no logged nonfinite loss |

### BDD RepPoints numerical-repair candidate

| Candidate / config | Completed steps / schedule | State |
|---|---:|---|
| [RepPoints R50 v3](../training/configs/bdd10k_ft_reppoints_r50_v3.py) | 10,900 / 18,000 | Training; epoch 8/12; no logged nonfinite loss so far |

The repair evaluates the classification focal loss in FP32 while retaining AMP elsewhere. The original FP16 kernel was reproduced producing nonfinite loss and gradients for finite, saturated logits. The old v2 checkpoints, logs, and results are preserved; v3 has not yet replaced v2 in any accepted result set.

The full repair run must complete all 18,000 finite-loss batches, pass final checkpoint checks, and finish independent validation before a success or accuracy-improvement claim. The inherited double normalization in the v2/v3 recipe remains a separately documented defect: it was deliberately not changed in this isolated numerical comparison. **This repair does not make the entire inherited recipe a recommended baseline.**

## The four data domains

Counts below were read from the actual COCO-format annotations used by these jobs. They describe this project's local splits, not necessarily the official benchmark splits. Counts are before training-time filtering; for example, Mobility excludes some empty/small images. The historical `10k` model-name prefix is an identifier, not a reliable image count.

| Domain | Train images | Train annotations | Validation images | Validation annotations | Classes |
|---|---:|---:|---:|---:|---:|
| BDD project split | 12,000 | 208,517 | 4,000 | 69,759 | 11 |
| Mobility / CMA | 8,456 | 21,850 | 784 | 1,580 | 5 |
| SeaDronesSee | 8,930 | 57,760 | 1,547 | 9,630 | 5 |
| WAID | 10,056 | 163,243 | 2,873 | 46,696 | 6 |

Class order in the corresponding training configs:

- **BDD project split:** person, rider, car, bus, truck, bike, motor, traffic light, traffic sign, train, animal. This is the project's 11-class conversion; do not label it the standard BDD100K detection protocol.
- **Mobility / CMA:** person, wheelchair, rollator, crutch, cane. See [the Mobility guide](../Mobility.md).
- **SeaDronesSee:** swimmer, boat, jetski, life_saving_appliances, buoy.
- **WAID:** sheep, cattle, seal, camelus, kiang, zebra.

This release includes aggregate split metadata and configuration references, not dataset images or per-image annotations. Obtain datasets separately under their applicable terms; no redistribution permission is implied.

## Evidence and limits

This snapshot was assembled by read-only inspection of local configuration files, training logs, `TRAIN_DONE` markers, checkpoint filenames, active trainer processes, and annotation counts. Raw private logs, filesystem paths, hostnames, and process identifiers are not part of the public snapshot.

A completion marker alone is insufficient. For each completed job, the selected run log also records the scheduled final checkpoint save and final validation. Mobility Deformable DETR uses its preserved original 24-epoch run log: a later unintended restart overwrote the convenience log but was stopped in epoch 1 without replacing the completed checkpoints.

“No logged nonfinite loss” has a narrow meaning: no `nan` or `inf` was found in loss fields in the selected periodic reporting windows. It is **not** a claim that every training batch was audited, that checkpoint tensors or optimizer states were revalidated for this publication snapshot, or that evaluation caches were cryptographically matched to their checkpoints. Gradient-norm overflow messages are distinct from nonfinite loss windows.

No mAP figures are published in this snapshot. Trainer validation output and older repository CSVs are not silently promoted into a newly verified accuracy table. Such a table needs an explicit evaluation protocol, class mapping, checkpoint digest, and prediction provenance.

Training progress is also not a latency or energy benchmark. No new cost measurements were taken for this update, and these training statuses establish no serving-SLO or hard-budget guarantee.
