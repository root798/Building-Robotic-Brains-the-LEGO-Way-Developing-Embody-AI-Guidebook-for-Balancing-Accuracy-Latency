"""Evaluate one checkpoint on explicit COCO annotations and image paths."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import types


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def apply_protocol(model_config):
    changed = []

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == 'score_thr' and isinstance(child, (int, float)):
                    value[key] = 0.001
                    changed.append(key)
                elif key == 'max_per_img' and isinstance(child, int):
                    value[key] = 300
                    changed.append(key)
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(model_config.get('test_cfg', {}))
    if not changed:
        model_config.setdefault('test_cfg', {})['max_per_img'] = 300
        changed.append('max_per_img(added)')
    queries = model_config.get('num_queries')
    if queries and queries < 300:
        model_config.setdefault('test_cfg', {})['max_per_img'] = int(queries)
        changed.append('max_per_img=num_queries({})'.format(queries))
    return sorted(set(changed))


def label_mapping(classes, categories):
    ordered = sorted(categories, key=lambda c: c['id'])
    if not classes:
        return {i: c['id'] for i, c in enumerate(ordered)}
    by_name = {c['name']: c['id'] for c in ordered}
    return {i: by_name.get(name) for i, name in enumerate(classes)}


def image_path(root, filename):
    root = Path(root).resolve()
    for sub in ('', 'val', 'test', 'train', '10k'):
        candidate = (root / sub / filename).resolve()
        if root not in candidate.parents:
            raise ValueError('Image filename escapes the supplied image root')
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(filename)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    for path in (args.config, args.checkpoint, args.annotations):
        if not path.is_file():
            parser.error('Missing input file: ' + str(path))
    if not args.images.is_dir():
        parser.error('Image root must be a directory')
    if args.output.exists() or args.output.is_symlink():
        parser.error('Use a new output directory')

    import torch
    import mmdet
    import mmengine
    from mmengine.config import Config
    from mmdet.apis import init_detector, inference_detector
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    inputs = {k: sha256(v) for k, v in [('config', args.config),
              ('checkpoint', args.checkpoint), ('annotations', args.annotations)]}
    cfg = Config.fromfile(str(args.config))
    applied = apply_protocol(cfg.model)
    detector = init_detector(cfg, str(args.checkpoint), device=args.device)
    head = getattr(detector, 'bbox_head', None)
    if head is not None and not hasattr(head, 'loss_cls'):
        head.loss_cls = types.SimpleNamespace(use_sigmoid=bool(getattr(head, 'use_sigmoid_cls', False)))
        applied.append('ssd_loss_cls_shim')
    coco = COCO(str(args.annotations))
    ids = sorted(coco.getImgIds())
    if not ids:
        raise ValueError('Annotation split contains no images')
    classes = tuple(detector.dataset_meta.get('classes', ()))
    mapping = label_mapping(classes, list(coco.cats.values()))
    predictions, skipped = [], set()
    args.output.mkdir(parents=True, exist_ok=False)
    for index, iid in enumerate(ids):
        path = image_path(args.images, coco.imgs[iid]['file_name'])
        with torch.no_grad():
            output = inference_detector(detector, str(path)).pred_instances
        for box, score, label in zip(output.bboxes.cpu().tolist(), output.scores.cpu().tolist(), output.labels.cpu().tolist()):
            category = mapping.get(int(label))
            if category is None:
                skipped.add(int(label))
                continue
            if not all(math.isfinite(x) for x in box + [score]):
                raise ValueError('Nonfinite prediction')
            x1, y1, x2, y2 = box
            predictions.append(dict(image_id=int(iid), category_id=category,
                                    bbox=[x1, y1, x2 - x1, y2 - y1], score=score))
        if index % 500 == 0:
            print('{}/{} images'.format(index, len(ids)), flush=True)
    prediction_file = args.output / 'predictions.json'
    prediction_file.write_text(json.dumps(predictions, allow_nan=False))
    stats = [None] * 6
    if predictions:
        evaluator = COCOeval(coco, coco.loadRes(str(prediction_file)), 'bbox')
        evaluator.params.imgIds = ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        stats = [float(x) for x in evaluator.stats[:6]]
    keys = ('bbox_mAP', 'mAP50', 'mAP75', 'mAP_s', 'mAP_m', 'mAP_l')
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  input_sha256=inputs, predictions_sha256=sha256(prediction_file),
                  n_images=len(ids), n_dets=len(predictions), metrics=dict(zip(keys, stats)),
                  protocol=dict(score_thr=0.001, max_per_img=300, applied=applied,
                                coco_max_dets=[1, 10, 100], batch=1, fp32=True, tta=False, scale='native'),
                  model_classes=list(classes), skipped_labels=sorted(skipped),
                  versions=dict(torch=torch.__version__, mmdet=mmdet.__version__, mmengine=mmengine.__version__))
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['metrics'], allow_nan=False))


if __name__ == '__main__':
    main()
