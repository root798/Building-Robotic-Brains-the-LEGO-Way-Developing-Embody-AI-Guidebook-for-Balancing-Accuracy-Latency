"""Model catalog and evaluation checks requiring only Python's stdlib."""
import ast
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


BUILD = module('build_model_zoo', 'tools/build_model_zoo.py')
EVAL = module('evaluate', 'evaluation/evaluate.py')


class ModelZooTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((ROOT / 'model_zoo/catalog.json').read_text())
        cls.rows = cls.catalog['models']

    def test_coverage_and_revision_counts(self):
        self.assertEqual(len(self.rows), 138)
        self.assertEqual(len({r['model'] for r in self.rows}), 138)
        evaluated = [r for r in self.rows if r['metrics'] is not None]
        self.assertEqual(len(evaluated), 136)
        self.assertEqual(Counter(r['domain'] for r in evaluated),
                         dict(bdd=34, mobility=32, seadronessee=35, waid=35))
        for domain in BUILD.DOMAINS:
            main = [r for r in self.rows if r['domain'] == domain and r['primary']]
            self.assertEqual(len(main), 32)
            self.assertEqual({r['architecture'] for r in main}, set(BUILD.NAMES))

    def test_configs_and_checkpoint_identities(self):
        for row in self.rows:
            path = ROOT / row['config']
            self.assertTrue(path.is_file())
            tree = ast.parse(path.read_text())
            if row['metrics'] is None:
                self.assertEqual(row['status'], 'training')
                self.assertIsNone(row['checkpoint'])
                continue
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row['config_sha256'])
            checkpoint = row['checkpoint']
            self.assertRegex(checkpoint['evaluated_sha256'], r'^[0-9a-f]{64}$')
            self.assertEqual(checkpoint['identity_matches'],
                             checkpoint['evaluated_sha256'] == checkpoint['available_file_sha256'])
            self.assertIsNone(checkpoint['download_url'])
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertFalse(node.value.startswith(('/', '~', 'file://')))

    def test_metrics_splits_and_protocol(self):
        sizes = {d['domain']: d['splits']['val']['images'] for d in self.catalog['datasets']}
        for row in self.rows:
            if row['metrics'] is None:
                continue
            self.assertEqual(row['n_images'], sizes[row['domain']])
            self.assertEqual(row['protocol']['max_per_img'], 300)
            self.assertEqual(row['protocol']['score_thr'], .001)
            self.assertTrue(row['protocol']['fp32'])
            self.assertFalse(row['protocol']['tta'])
            for metric in row['metrics'].values():
                if metric is not None:
                    self.assertGreaterEqual(metric, -1)
                    self.assertLessEqual(metric, 1)
        self.assertNotIn('NaN', json.dumps(self.catalog, allow_nan=False))

    def test_known_issues_are_preserved(self):
        by_id = {r['model']: r for r in self.rows}
        self.assertEqual(by_id['bdd10k_ft_reppoints_r50_v2']['status'], 'numerical_issue')
        self.assertEqual(by_id['bdd10k_ft_diffusiondet_r50']['status'], 'checkpoint_mismatch')
        self.assertFalse(by_id['bdd10k_ft_reppoints_r50_v3']['primary'])
        self.assertTrue(by_id['waid10k_ft_vitdet_b']['primary'])
        for model in ('bdd10k_ft_efficientdet_d3', 'bdd10k_ft_reppoints_r50'):
            self.assertEqual(by_id[model]['status'], 'no_detections')
            self.assertIsNone(by_id[model]['metrics']['bbox_mAP'])

    def test_markdown_and_csv_match_catalog(self):
        for domain, (_, _, _, page) in BUILD.DOMAINS.items():
            self.assertEqual((ROOT / page).read_text(), BUILD.render_domain(self.catalog, domain))
        with (ROOT / 'model_zoo/results.csv').open() as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(len(csv_rows), len(self.rows))
        lookup = {r['model']: r for r in self.rows}
        for flat in csv_rows:
            row = lookup[flat['model']]
            self.assertEqual(flat['status'], row['status'])
            for key in BUILD.METRICS:
                expected = (row['metrics'] or {}).get(key)
                actual = float(flat[key]) if flat[key] else None
                self.assertEqual(actual, expected)

    def test_selection_does_not_depend_on_ap(self):
        low = dict(model='x_frcnn_r50_fpn', metrics=dict(bbox_mAP=.01))
        high = dict(model='x_faster_rcnn', metrics=dict(bbox_mAP=.99))
        self.assertIs(BUILD.preferred([low, high]), low)
        replacement = dict(model='x_v2', metrics=dict(bbox_mAP=.0))
        self.assertIs(BUILD.preferred([high, replacement]), replacement)

    def test_path_only_export(self):
        source = "data_root = '/private/Dataset/mobility/'\nload_from = '/weights/base.pth'\nwork_dir = '/old/work'\nmodel = dict(depth=50)\n"
        result = BUILD.portable_config(source, 'example')
        self.assertIn("data_root = 'data/Dataset/mobility/'", result)
        self.assertIn("load_from = 'checkpoints/base.pth'", result)
        self.assertIn('model = dict(depth=50)', result)
        with self.assertRaises(ValueError):
            BUILD.portable_config("unknown = '/private/unknown'", 'example')


class EvaluationTest(unittest.TestCase):
    def test_nested_thresholds_and_query_cap(self):
        config = dict(test_cfg=dict(rcnn=dict(score_thr=.5, max_per_img=100)))
        EVAL.apply_protocol(config)
        self.assertEqual(config['test_cfg']['rcnn'], dict(score_thr=.001, max_per_img=300))
        config = dict(num_queries=100)
        EVAL.apply_protocol(config)
        self.assertEqual(config['test_cfg']['max_per_img'], 100)

    def test_class_name_mapping(self):
        cats = [dict(id=0, name='person'), dict(id=4, name='cane')]
        self.assertEqual(EVAL.label_mapping(('cane', 'unknown', 'person'), cats), {0: 4, 1: None, 2: 0})
        self.assertEqual(EVAL.label_mapping((), cats), {0: 0, 1: 4})

    def test_image_resolution_and_traversal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'val').mkdir()
            (root / 'val/image.jpg').touch()
            self.assertEqual(EVAL.image_path(root, 'image.jpg'), root / 'val/image.jpg')
            with self.assertRaises(ValueError):
                EVAL.image_path(root, '../other.jpg')

    def test_cli_help_without_ml_imports(self):
        result = subprocess.run([sys.executable, '-m', 'evaluation.evaluate', '--help'], cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--checkpoint', result.stdout)


if __name__ == '__main__':
    unittest.main()
