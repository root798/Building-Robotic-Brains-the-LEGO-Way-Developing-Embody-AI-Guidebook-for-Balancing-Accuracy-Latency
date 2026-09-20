"""Train one trusted MMDetection config in a strictly new output directory.

Run from the repository root with ``python -m training.train CONFIG --work-dir
OUTPUT``. Device visibility and resource allocation remain the caller's choice.
This entry point never writes TRAIN_DONE or launches subsequent evaluation.
"""

import argparse
import os
from pathlib import Path
import types


def reserve_work_dir(path):
    """Atomically refuse any existing output path, including dangling symlinks."""
    output = Path(path).expanduser().absolute()
    if os.path.lexists(str(output)):
        raise FileExistsError("Refusing existing work directory: %s" % output)
    # exist_ok=False also closes the race between the check and creation.
    output.mkdir(parents=True, exist_ok=False)
    return output


def add_ssd_loss_compatibility(model):
    """Preserve the historical MMDetection 3.3 SSD/AnchorHead compatibility shim."""
    head = getattr(model, "bbox_head", None)
    if head is not None and not hasattr(head, "loss_cls"):
        head.loss_cls = types.SimpleNamespace(
            use_sigmoid=bool(getattr(head, "use_sigmoid_cls", False)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Trusted MMDetection Python config")
    parser.add_argument(
        "--work-dir", required=True,
        help="New output directory; even an existing empty directory is refused")
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        parser.error("Config is not a file: %s" % config_path)
    output_path = Path(args.work_dir).expanduser().absolute()
    if os.path.lexists(str(output_path)):
        parser.error("Refusing existing work directory: %s" % output_path)

    # Keep --help and argument checks usable without importing the GPU stack.
    # Python configs execute code: only load configs you trust.
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(str(config_path))
    if cfg.get("resume", False):
        parser.error("This fresh-run entry point refuses configs with resume=True")
    cfg.resume = False
    try:
        cfg.work_dir = str(reserve_work_dir(output_path))
    except FileExistsError as error:
        parser.error(str(error))
    runner = Runner.from_cfg(cfg)
    add_ssd_loss_compatibility(runner.model)
    runner.train()
    # A hook's finite-loss certificate is only one acceptance check. Never turn
    # process success into a TRAIN_DONE marker without checkpoint validation.


if __name__ == "__main__":
    main()
