# Reproducing a training recipe

This release contains 20 campaign configurations and one separately identified
RepPoints numerical-repair candidate. It publishes source and dated status,
not datasets, checkpoints, a complete environment lock or a claim that every
recipe works unchanged in a fresh upstream installation.

## Environment boundary

The historical experiment environment reports these core versions:

| Component | Historical version |
| --- | --- |
| Python | 3.8 |
| PyTorch | 1.10.2 |
| MMCV | 2.1.0, with compiled operators |
| MMEngine | 0.10.7 |
| MMDetection | 3.3.0 checkout |

These versions are an environment record, **not a tested installation lock**.
CUDA, compiler, compiled MMCV operators, additional packages and checkout-local
compatibility changes matter. The complete historical dependency tree is not
vendored or hash-locked here. In particular, historical legacy transforms and
project models must be available in the chosen environment. Validate imports
and data loading before allocating a full training run; do not interpret
configuration parsing as a successful end-to-end reproduction.

Use an isolated environment you control. Do not upgrade packages in another
user's environment to make a recipe load. ViTDet and DiffusionDet configurations
retain their `projects.ViTDet.vitdet` and `projects.DiffusionDet.diffusiondet`
imports; provide the corresponding compatible MMDetection project modules.
The [configuration manifest](../training/configs/manifest.json) records each
custom import and required checkpoint basename.

## External assets and paths

Run commands from the repository root. Published paths are relative:

| Domain | Expected dataset root | Training annotations under that root |
| --- | --- | --- |
| BDD100K-FT | `data/bdd100k_ft/` | `annotations/bdd10k_ft_train.json` |
| Mobility / CMA | `data/Dataset/mobility_coco/` | `annotations/instances_train.json` |
| SeaDronesSee | `data/Dataset/SeaDronesSeeObjectDetectionv2/mmdet_format/` | `annotations/instances_train.json` |
| WAID | `data/Dataset/WAID/WAID/` | `annotations_mmdet/instances_train.json` |

Obtain images, COCO-format annotations and initialization weights through their
authorized sources and respect their terms. Supply the exact split, category
names/order and image paths required by your chosen configuration. The release
does not ship a conversion/split-generation pipeline or verify independently
downloaded files against the original private data. Data paths may be directory
links to assets you own; do not move or relabel someone else's files.

Initialization weights belong in `checkpoints/` using the manifest's basenames.
Check all `load_from` and `init_cfg.checkpoint` fields: some models require more
than one file. Filename agreement alone does not prove checkpoint identity;
record SHA256 digests for your own assets and published results. The configured
model and domain class count must agree with the data, even when a COCO
initialization head is replaced during fine-tuning.

The exporter changes storage paths and the repair's module import, and removes
private absolute provenance paths. It does not silently adjust the inherited
optimization or preprocessing recipes. Original source and exported-file
hashes are recorded in the manifest.

## Checks without GPU work

Publication and entry-point checks require only Python's standard library:

```bash
python -m unittest discover -s tests -p 'test_training_configs.py'
python -m unittest discover -s tests -p 'test_publication.py'
python -m unittest discover -s tests -p 'test_train_cli.py'
python -m training.train --help
```

The six finite-loss hook tests additionally require the compatible PyTorch,
MMEngine, MMCV and MMDetection environment, but allocate no CUDA tensors:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m unittest discover -s tests -p 'test_finite_loss_hook.py' -v
```

These tests exercise mocked training batches. They do not load a dataset or
checkpoint and do not certify model accuracy. Do not add the manual CUDA test
to unattended CPU CI.

## One fresh training run

Before running, check your machine's resource policy and obtain an idle device
you are authorized to use. The following example selects device `0` only as
an example; it does not assert that device `0` is available on your machine.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -m training.train \
  training/configs/bdd10k_ft_reppoints_r50_v3.py \
  --work-dir work_dirs/bdd_reppoints_v3_attempt_001
```

Python configuration files execute code: only load configurations you trust.
The [entry point](../training/train.py) requires a new `--work-dir`, refusing
even an existing empty directory or dangling symlink. It disables automatic
resume, refuses `resume=True`, preserves the historical SSD compatibility
shim, and delegates one run to MMEngine. Failures retain their output directory
for inspection; choose a new name for a subsequent attempt.

The launcher neither auto-selects/reserves GPUs nor stops other processes. It
does not alter system packages, clocks, power settings or scheduling policy.
Apply CPU affinity and priority according to your own shared-machine policy.
It does not queue more experiments, run independent detection regeneration,
write `TRAIN_DONE`, or promote results. Any validation built into the chosen
MMEngine training configuration still executes normally.

The numerical-repair configuration enables the finite-loss hook; the other
historical configurations are not silently modified to enable it. Read
[numerical stability](numerical-stability.md) before interpreting its output
certificate, especially the unresolved duplicate-normalization caveat.

## Optional manual CUDA reproduction

Only after choosing an authorized idle GPU, run:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m training.test_focal_cuda --run-cuda
```

This test allocates GPU memory and deliberately evaluates the old failing
kernel on synthetic inputs before checking the repair. It is a numerical
regression, not a model benchmark or timing measurement. Results depend on
the compiled kernel version; a newer environment may not reproduce the
historical failure.

## Evidence before a completion claim

Preserve the exact configuration, environment record, asset hashes, full log,
checkpoint identity and evaluation protocol. Check expected epochs/iterations,
loss history, checkpoint/optimizer tensors and validation independently. A
zero exit code, existing checkpoint or completion marker alone is insufficient.
For the BDD repair, the target is 12 epochs and 18,000 checked batches; the
finite-loss certificate is necessary but not sufficient.

The legacy domain pages and performance CSVs are historical references, not
the validation outputs of this campaign. Built-in validation and a separately
unified evaluation protocol can differ in detection thresholds/caps, so do
not join their metrics without identifying the protocol. No clean inference
latency, energy or serving-budget claim follows from these training records.
