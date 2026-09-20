# Third-party notices

The detector recipes use and derive from the OpenMMLab MMDetection ecosystem.
Original attribution: **Copyright 2018–2023 OpenMMLab. All rights reserved.**
The upstream [MMDetection v3.3.0 license](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/LICENSE)
is retained in [licenses/MMDetection-Apache-2.0.txt](licenses/MMDetection-Apache-2.0.txt).

Published configurations are modified project recipes: domain-specific classes,
dataset blocks, training schedules and numerical repairs are recorded in their
provenance. This publication further relocates machine-specific paths and adds
export hashes. These files are not unchanged upstream configurations or a new
claim of ownership over the underlying detector implementations.

MMCV, MMEngine, PyTorch and the optional MMDetection project implementations are
external dependencies, not vendored here. Consult their respective upstream
notices when installing or redistributing them. The small numerical adapter
calls the existing focal-loss API; it does not include the CUDA kernel source.

Dataset citations are retained on the domain pages and in the main README.
No images, annotations or pretrained/fine-tuned weights are distributed by this
training update, and no right to redistribute those assets is asserted.

This notice preserves upstream rights. It does not relicense historical files,
select a repository-wide license, or change dataset terms.
