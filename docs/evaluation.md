# Evaluation protocol

The model zoo reports **single-checkpoint, full-validation COCO bbox AP**.
JSON and CSV store AP in 0–1 units; domain pages display 0–100.

| Domain | Split | Images | Categories |
| --- | --- | ---: | ---: |
| BDD100K-FT | Project validation | 4,000 | 11 |
| Mobility / CMA | Validation | 784 | 5 |
| SeaDronesSee | Validation | 1,547 | 5 |
| WAID | Validation | 2,873 | 6 |

Each model uses its native configuration test scale, FP32 inference, batch size
1, and no test-time augmentation. Existing score thresholds are set to 0.001
and prediction caps to 300 where supported; DETR query counts can impose a
smaller cap. The catalog records the adjustments applied to each evaluation.
COCOeval uses its default **maxDets = [1, 10, 100]**, distinct from the stored
prediction cap. AP averages IoU thresholds 0.50–0.95; AP50 and AP75 use a
single IoU threshold.

Checkpoint labels are mapped to annotation categories by class name.
Unmatched categories are omitted and recorded. In particular, the BDD TOOD
checkpoint has a different class vocabulary; its catalog entry records the
mapping. These are project splits, not an official cross-benchmark leaderboard.

## Catalog structure

Each evaluated row records its recipe, six AP metrics, evaluated checkpoint
SHA-256, available checkpoint identity check, source metadata hash, class
mapping, framework versions, and evaluation date. The configuration source and
portable export have separate hashes. These recipe snapshots were exported at
publication time; historical evaluation metadata did not hash its config file.
Raw per-image predictions and timing logs are not part of the publication.

There are 32 architecture slots per domain. A newer v2 run replaces the earlier
recipe in the main table. For Faster R-CNN aliases, the explicit `frcnn_r50_fpn`
run is selected; the small-object FreeAnchor variant is kept separately.
Selection does not use validation AP. Other revisions remain in the catalog
and each domain’s secondary table.

These are individual trained checkpoints, not means over repeated seeds.
No-detection evaluations retain missing AP as `null`. Numerical issues and
checkpoint availability are marked in the tables; details are in
[known issues](known-issues.md). Older metadata that did not record class
vocabularies or skipped-label lists retains `null` for those fields.

## Re-evaluation

Use [`evaluation/evaluate.py`](../evaluation/evaluate.py) with explicit config,
checkpoint, annotation, image-root, and output paths. It applies the protocol
above, writes predictions plus input hashes, and preserves missing metrics for
an empty prediction set. See [commands](reproduction.md).
