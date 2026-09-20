"""Manual GPU regression for the historical MMCV FP16 focal-loss failure.

This is not part of CPU test discovery. It allocates CUDA tensors only when
explicitly invoked with ``python -m training.test_focal_cuda --run-cuda``.
The caller must choose an idle, authorized GPU through CUDA_VISIBLE_DEVICES.
"""

import argparse
import json
import os


def evaluate(criterion, values, labels, weights, dtype):
    import torch

    pred = values.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    with torch.cuda.amp.autocast():
        loss = criterion(pred, labels, weight=weights, avg_factor=4.)
    loss.backward()
    return loss.detach(), pred.grad.detach()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-cuda", action="store_true", help="Authorize this GPU test")
    args = parser.parse_args(argv)
    if not args.run_cuda:
        parser.error("Manual GPU test: pass --run-cuda after selecting an idle GPU")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() in ("", "-1"):
        parser.error("Set CUDA_VISIBLE_DEVICES explicitly to an authorized, idle GPU")

    import torch
    from mmdet.models.losses import FocalLoss
    from training.reppoints_numerics import Float32FocalLoss, _nonfinite_paths

    if not torch.cuda.is_available():
        parser.error("No usable CUDA device is visible")
    torch.manual_seed(0)
    labels = torch.tensor([0, 5, 10, 11], device="cuda", dtype=torch.long)
    weights = torch.tensor([1., 1., 1., 0.], device="cuda")
    original = FocalLoss()
    fixed = Float32FocalLoss()
    results = []
    for logit in (-100., -32., -20., -10., 0., 10., 20., 32., 100.):
        values = torch.full((4, 11), logit)
        old, old_grad = evaluate(original, values, labels, weights, torch.float16)
        repaired, repaired_grad = evaluate(fixed, values, labels, weights, torch.float16)
        reference, reference_grad = evaluate(original, values, labels, weights, torch.float32)
        assert torch.isfinite(repaired).all() and torch.isfinite(repaired_grad).all()
        assert repaired.dtype == torch.float32 and repaired_grad.dtype == torch.float16
        assert torch.allclose(repaired, reference, atol=1e-6, rtol=1e-6)
        assert torch.allclose(repaired_grad.float(), reference_grad, atol=1e-3, rtol=1e-3)
        results.append(dict(logit=logit, original_finite=bool(
            torch.isfinite(old).all() and torch.isfinite(old_grad).all()),
            repaired_finite=True, matches_float32=True))

    values = (torch.rand(4, 11) * 8. - 4.).half().float()
    old, old_grad = evaluate(original, values, labels, weights, torch.float32)
    repaired, repaired_grad = evaluate(fixed, values, labels, weights, torch.float16)
    assert torch.allclose(repaired, old, atol=1e-6, rtol=1e-6)
    assert torch.allclose(repaired_grad.float(), old_grad, atol=1e-3, rtol=1e-3)
    assert _nonfinite_paths({"loss_cls": [torch.tensor(float("nan"))]}, "losses") == [
        "losses.loss_cls[0]"]
    assert not _nonfinite_paths({"loss_cls": [torch.tensor(1.)]}, "losses")
    assert any(not r["original_finite"] for r in results), (
        "Original failure was not reproduced in this environment; inspect versions "
        "instead of claiming reproduction of the historical kernel bug")
    print(json.dumps(dict(status="PASS", saturation_cases=results,
                          ordinary_logits_match=True, finite_guard_detects_nan=True), indent=2))


if __name__ == "__main__":
    main()
