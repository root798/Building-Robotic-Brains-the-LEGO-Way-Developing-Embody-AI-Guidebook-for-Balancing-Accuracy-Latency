"""Project-local numerical safeguards; no edits to MMDetection are required."""

import json
from pathlib import Path

import torch
from mmdet.models.losses import FocalLoss
from mmdet.registry import MODELS
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@MODELS.register_module()
class Float32FocalLoss(FocalLoss):
    """Run focal loss and its CUDA backward in FP32, even with an AMP head.

    MMCV 2.1's half-precision kernel casts FLT_MIN to half (zero), allowing
    log(0) * 0 at sigmoid saturation. Cast BEFORE the kernel, never after it.
    This preserves labels, weighting, reduction, gamma, alpha and autograd.
    """

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None):
        target = target.float() if target.is_floating_point() else target
        if weight is not None:
            weight = weight.float()
        if isinstance(avg_factor, torch.Tensor):
            avg_factor = avg_factor.float()
        with torch.cuda.amp.autocast(enabled=False):
            return super().forward(
                pred.float(), target, weight=weight, avg_factor=avg_factor,
                reduction_override=reduction_override)


def _nonfinite_paths(value, path=""):
    """Return precise loss component paths, including multi-level loss lists."""
    if isinstance(value, torch.Tensor):
        return [path] if not bool(torch.isfinite(value).all()) else []
    if isinstance(value, dict):
        return [p for key, item in value.items()
                for p in _nonfinite_paths(item, path + "." + key)]
    if isinstance(value, (list, tuple)):
        return [p for i, item in enumerate(value)
                for p in _nonfinite_paths(item, path + "[%d]" % i)]
    return []


@HOOKS.register_module()
class FiniteTrainingLossHook(Hook):
    """Fail before an optimizer update if any raw loss component is nonfinite.

    A successful process exit must not silently label a run with NaN loss as
    TRAIN_DONE. AMP gradient-scaler warm-up skips are distinct from a NaN loss
    and are deliberately not treated as this failure condition.
    """

    priority = "VERY_HIGH"

    def before_train(self, runner):
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        original_loss = model.loss
        self.original_loss = original_loss
        self.model = model
        self.checked_batches = 0
        self.start_iteration = int(runner.iter)

        def checked_loss(*args, **kwargs):
            losses = original_loss(*args, **kwargs)
            bad = _nonfinite_paths(losses, "losses")
            if bad:
                samples = kwargs.get("batch_data_samples")
                if samples is None and len(args) > 1:
                    samples = args[1]
                event = dict(
                    epoch=int(runner.epoch) + 1, iteration=int(runner.iter) + 1,
                    components=bad,
                    image_ids=[sample.metainfo.get("img_id")
                               for sample in (samples or [])])
                path = Path(runner.work_dir) / "NONFINITE_LOSS.json"
                path.write_text(json.dumps(event, indent=2) + "\n")
                raise FloatingPointError("Nonfinite training loss before update: %s" % event)
            self.checked_batches += 1
            return losses

        model.loss = checked_loss

    def after_train(self, runner):
        self.model.loss = self.original_loss
        if (int(runner.iter) != int(runner.max_iters) or
                self.checked_batches != int(runner.iter) - self.start_iteration):
            raise RuntimeError("Training ended before all expected batches were checked")
        result = dict(checked_batches=self.checked_batches,
                      nonfinite_loss_batches=0, completed=True,
                      start_iteration=self.start_iteration,
                      final_iteration=int(runner.iter), final_epoch=int(runner.epoch))
        path = Path(runner.work_dir) / "FINITE_TRAINING.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
