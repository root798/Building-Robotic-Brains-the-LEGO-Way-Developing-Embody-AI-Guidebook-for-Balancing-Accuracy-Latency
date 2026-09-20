# RepPoints numerical stability

This is a dated diagnosis and targeted repair, not a claim of improved accuracy.
At the 2026-09-20 publication snapshot, the full BDD RepPoints repair run is
still in progress. Consult [training status](training-status.md) for the dated
inventory; a later successful unit test does not update that inventory.

## What failed

The historical BDD RepPoints v2 run has 23 NaN reporting windows among 360
logged windows. All affected windows report a nonfinite classification loss;
both point-regression losses remain finite. The first affected window is epoch
6, iteration 1,300 within that epoch. A reporting window is not an exact count
of bad optimizer steps.

The last three inspected checkpoints have finite model weights, and the last
two have finite optimizer momentum. Nevertheless, the final AMP loss scale is
0.0009765625. Dynamic loss scaling can skip updates after nonfinite gradients,
so finite weights and an exit code of zero do not establish healthy training.

The installed MMCV 2.1 CUDA focal-loss kernel receives half-precision logits.
It casts `FLT_MIN` to the input type; that floor becomes zero in FP16. When
the sigmoid saturates, the kernel's masked expressions can evaluate
`log(0) * 0`, producing NaN in the forward and backward calculations. Lowering
the learning rate or the AMP loss scale does not repair this forward-kernel
operation. This diagnosis is specific to the reproduced classification-loss
failure, not every historical divergence or every newer MMCV build.

## Isolated repair

[`Float32FocalLoss`](../training/reppoints_numerics.py) disables autocast inside
the classification loss and casts logits **before** entering the loss kernel.
Its loss calculation and CUDA backward use FP32; gradients remain connected
to the original AMP head. Labels, weights, reduction, gamma and alpha retain
the original focal-loss semantics.

The [v3 configuration](../training/configs/bdd10k_ft_reppoints_r50_v3.py) inherits
the [v2 recipe](../training/configs/bdd10k_ft_reppoints_r50_v2.py), adding the
FP32 loss, finite-loss guard, module import and distinct output identity. Batch
size 8, SGD learning rate 0.005, gradient clipping at 35, seed 0, COCO
initialization, data transforms and the 12-epoch schedule are unchanged. The
candidate starts a fresh run; it does not resume a checkpoint with skipped
updates. Historical results remain separate from the candidate.

## Regression evidence

The historical GPU reproduction compared identical logits with the original
FP16 loss, the repair, and the original FP32 reference. In nine cases with
constant logits `-100, -32, -20, -10, 0, 10, 20, 32, 100`, the repair returned
finite losses and gradients and matched the FP32 reference. The original FP16
path was nonfinite in seven of the nine cases; `-10` and `0` remained finite.
An additional ordinary-logit comparison also matched. Loss tolerance is
`atol=rtol=1e-6`; gradient tolerance is `atol=rtol=1e-3` after the gradient's
cast back to FP16.

The [manual CUDA regression](../training/test_focal_cuda.py) publishes this
same-input check. It was not rerun on occupied GPUs for this repository update.
It deliberately requires both explicit device visibility and `--run-cuda`.
If a different environment no longer reproduces the original failure, the
test reports that rather than claiming reproduction of the historical bug.

Six [CPU guard tests](../tests/test_finite_loss_hook.py) pass using the real
MMEngine `BaseModel.train_step` dispatch, with only the detector, optimizer
actions and artifact writes mocked:

1. A finite batch reaches the optimizer and is counted.
2. A NaN batch is rejected before the optimizer update, with its component
   path and image ID recorded.
3. Early termination cannot produce a completion certificate.
4. A fully checked run produces the certificate and restores the loss method.
5. An unchecked iteration cannot produce a completion certificate.
6. Resumed-run accounting counts only the newly checked iterations.

The guard records `NONFINITE_LOSS.json` and raises on a nonfinite raw loss
component. `FINITE_TRAINING.json` is written only when the runner reaches its
expected iteration count and every new iteration was checked. The `completed`
field means that this finite-loss check completed, **not** that checkpoint
integrity, validation accuracy, preprocessing or scientific claims passed.
The current public entry point supports fresh single-process runs, not resume
or distributed-training certification.

Initial `grad_norm: inf` during dynamic AMP scale calibration is distinct from
a nonfinite raw loss. The guard does not certify that every gradient or
optimizer tensor is finite. Acceptance of the actual BDD candidate additionally
requires all 18,000 expected batches, finite final checkpoint and optimizer
tensors, and separate native-resolution validation. The public launcher does
not manufacture `TRAIN_DONE` from process success.

## Separate recipe defect: deliberately not hidden

The inherited BDD RepPoints v2 and v3 pipelines apply both dataset
`Normalize(to_rgb=True)` and model `DetDataPreprocessor(bgr_to_rgb=True)` with
the same mean and standard deviation. CPU probing confirmed double
normalization and two channel reversals. The missing explicit training sampler
also resolves to sequential sampling in the historical environment.

Neither behavior was changed in this numerical-only comparison. Correcting
preprocessing or sampling requires a separately identified recipe and new
training comparison. The exported configurations preserve these limitations
for provenance; they are not endorsed as optimal recipes. Do not infer that
the repair improves mAP, that the whole campaign is clean, or that it changes
any frozen detector-pool or latency/energy result.

See the [reproduction guide](reproduction.md) for dependencies, safe commands
and external-asset requirements.
