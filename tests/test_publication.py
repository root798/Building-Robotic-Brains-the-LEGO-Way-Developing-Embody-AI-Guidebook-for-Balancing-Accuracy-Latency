"""Standard-library checks for public evidence, links and artifact boundaries."""

import ast
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
NEW_TREES = ('training', 'docs', 'tools', 'tests', '.github', 'licenses')
PUBLIC_ROOT_FILES = ('README.md', 'Mobility.md', 'CONTRIBUTING.md',
                     'THIRD_PARTY_NOTICES.md', '.gitignore')


def publication_files():
    paths = [ROOT / name for name in PUBLIC_ROOT_FILES]
    for directory in NEW_TREES:
        paths.extend(p for p in (ROOT / directory).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts)
    return sorted(set(paths))


class PublicationTest(unittest.TestCase):
    def setUp(self):
        self.status = json.loads((ROOT / 'training/status.json').read_text())

    def test_inventory_is_dated_and_does_not_double_count_repair(self):
        datetime.fromisoformat(self.status['as_of_utc'].replace('Z', '+00:00'))
        rows = self.status['models']
        self.assertEqual(len({r['model'] for r in rows}), len(rows))
        formal = [r for r in rows if r['formal_job']]
        repairs = [r for r in rows if not r['formal_job']]
        self.assertEqual(len(formal), 20)
        self.assertEqual(len(repairs), 1)
        self.assertEqual(self.status['summary']['formal_jobs'], len(formal))
        self.assertEqual(Counter(r['domain'] for r in formal),
                         dict(mobility=9, seadronessee=5, waid=4, bdd=2))
        for state, count in Counter(r['status'] for r in formal).items():
            self.assertEqual(self.status['summary'][state], count)

    def test_completion_is_not_inferred_from_a_marker_alone(self):
        for row in self.status['models']:
            with self.subTest(model=row['model']):
                self.assertGreater(row['total_steps'], 0)
                self.assertGreaterEqual(row['completed_steps'], 0)
                self.assertLessEqual(row['completed_steps'], row['total_steps'])
                self.assertTrue((ROOT / row['config']).is_file())
                if row['status'] == 'completed_log_no_nonfinite_loss':
                    self.assertTrue(row['train_done'])
                    self.assertTrue(row['checkpoint_files_present'])
                    self.assertTrue(row['final_schedule_save_logged'])
                    self.assertTrue(row['final_validation_logged'])
                    self.assertEqual(row['completed_steps'], row['total_steps'])
                    self.assertEqual(row['logged_nonfinite_loss_windows'], 0)
                elif row['status'] == 'completed_with_nonfinite_loss':
                    self.assertGreater(row['logged_nonfinite_loss_windows'], 0)
                elif row['status'] == 'training':
                    self.assertTrue(row['live_trainer_observed'])
                    self.assertFalse(row['train_done'])

    def test_domain_category_counts(self):
        domains = {d['domain']: d for d in self.status['domains']}
        self.assertEqual(set(domains), {'bdd', 'mobility', 'seadronessee', 'waid'})
        for domain, categories in dict(bdd=11, mobility=5, seadronessee=5, waid=6).items():
            self.assertEqual(len(domains[domain]['classes']), categories)
            for split in ('train', 'val'):
                self.assertGreater(domains[domain]['splits'][split]['images'], 0)

    def test_publication_contains_no_machine_paths_or_secret_material(self):
        patterns = [r'/' + r'(?:home|media|mnt)/',
                    r'gh[pousr]_[A-Za-z0-9]{20,}',
                    r'github_pat_[A-Za-z0-9_]{20,}',
                    r'-----BEGIN ' + r'(?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
        forbidden = {'.pth', '.pt', '.ckpt', '.onnx', '.engine', '.safetensors'}
        for path in publication_files():
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertNotIn(path.suffix, forbidden)
                self.assertLess(path.stat().st_size, 500_000)
                text = path.read_text(encoding='utf-8')
                for pattern in patterns:
                    self.assertIsNone(re.search(pattern, text), pattern)

    def test_new_documentation_local_links_resolve(self):
        markdown = [p for p in publication_files() if p.suffix == '.md']
        for path in markdown:
            for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
                target = target.strip('<>').split('#', 1)[0]
                if not target or re.match(r'[a-z]+:', target):
                    continue
                with self.subTest(document=path.name, target=target):
                    self.assertTrue((path.parent / target).exists())

    def test_python_sources_parse_without_importing_training(self):
        for path in publication_files():
            if path.suffix == '.py':
                with self.subTest(path=path.name):
                    ast.parse(path.read_text(), filename=str(path))


if __name__ == '__main__':
    unittest.main(verbosity=2)
