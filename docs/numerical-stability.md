# RepPoints numerical stability

## Issue and repair

BDD RepPoints v2 recorded nonfinite classification losses. In the reproduced
MMCV 2.1 CUDA focal-loss path, casting `FLT_MIN` to FP16 makes the floor zero;
saturated sigmoid inputs can then evaluate masked `log(0) * 0` expressions.

[`Float32FocalLoss`](../training/reppoints_numerics.py) casts logits before the
loss kernel and disables autocast for that calculation, while preserving AMP
in the rest of the detector. The [v3 recipe](../training/configs/bdd10k_ft_reppoints_r50_v3.py)
keeps v2’s initialization, batch size 8, SGD learning rate 0.005, clipping at
35, seed 0, and 12-epoch schedule.

## Tests

The manual CUDA regression compares the repaired loss and gradients to the
FP32 reference on nine saturated-logit cases and an ordinary-logit case.
The repair was finite in all nine saturation cases; the original FP16 path
was nonfinite in seven. Six CPU tests exercise the per-batch guard, early
termination, complete-run accounting, and rejection before optimizer updates.

The guard writes `NONFINITE_LOSS.json` on failure and `FINITE_TRAINING.json`
after all expected iterations have been checked. Final checkpoint/optimizer
checks and a separate validation pass complete the repair’s acceptance process.
AMP gradient-scale calibration is distinct from a nonfinite raw loss.

See [training progress](training-status.md) for the dated run state. Accuracy
improvement is assessed only after the replacement evaluation finishes.
The inherited preprocessing issue remains separate; see [known issues](known-issues.md).

Commands are in the [reproduction guide](reproduction.md).
