"""CPU-only publication checks; no ML framework, dataset or GPU is imported."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_training_configs", ROOT / "tools" / "export_training_configs.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class PublishedConfigsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_dir = ROOT / "training" / "configs"
        cls.manifest = json.loads((cls.config_dir / "manifest.json").read_text())
        cls.records = {item["model"]: item for item in cls.manifest["configs"]}

    def test_explicit_scope(self):
        self.assertEqual(set(self.records), set(dict(EXPORT.ALL_JOBS)))
        self.assertEqual(len(self.records), 21)
        self.assertEqual(self.manifest["formal_job_count"], 20)
        self.assertEqual(self.manifest["repair_candidate_count"], 1)
        self.assertEqual(
            {path.stem for path in self.config_dir.glob("*.py")}, set(self.records))
        self.assertFalse(self.manifest["checkpoints_included"])
        self.assertFalse(self.manifest["data_included"])

    def test_each_export_parses_is_public_and_matches_hash(self):
        for model, record in self.records.items():
            with self.subTest(model=model):
                payload = (self.config_dir / record["published_filename"]).read_bytes()
                source = payload.decode("utf-8")
                tree = ast.parse(source)
                EXPORT.assert_public_source(source)
                self.assertEqual(hashlib.sha256(payload).hexdigest(), record["published_sha256"])
                self.assertRegex(record["source_sha256"], r"^[0-9a-f]{64}$")
                self.assertEqual(record["source_filename"], model + ".py")
                self.assertEqual(
                    ast.literal_eval(EXPORT.assignment(tree, "work_dir")),
                    "work_dirs/" + model)
                self.assertTrue(source.startswith(EXPORT.HEADER))
        self.assertIsNone(EXPORT.PRIVATE_PATH.search(json.dumps(self.manifest)))

    def test_repair_is_separate_and_inherits_v2(self):
        model = EXPORT.REPAIR_JOB[0]
        tree = ast.parse((self.config_dir / (model + ".py")).read_text())
        self.assertEqual(ast.literal_eval(EXPORT.assignment(tree, "_base_")),
                         "./bdd10k_ft_reppoints_r50_v2.py")
        self.assertEqual(self.records[model]["role"], "repair_candidate")
        self.assertEqual(self.records[model]["custom_imports"], ["training.reppoints_numerics"])
        self.assertEqual(self.records["bdd10k_ft_reppoints_r50_v2"]["role"], "formal_queue")

    def test_historical_transforms_and_sampler_are_not_silently_fixed(self):
        # v2 is an auditable recipe snapshot, not a claim of ideal preprocessing.
        tree = ast.parse((self.config_dir / "bdd10k_ft_reppoints_r50_v2.py").read_text())
        values = [node.value for node in EXPORT.strings(tree)]
        self.assertIn("Normalize", values)
        self.assertIn("DetDataPreprocessor", values)
        loader = EXPORT.assignment(tree, "train_dataloader")
        self.assertIsInstance(loader, ast.Call)
        self.assertNotIn("sampler", {item.arg for item in loader.keywords})

    def test_external_architecture_imports_are_retained(self):
        expected = {
            "mobility10k_ft_diffusiondet_r50": "projects.DiffusionDet.diffusiondet",
            "bdd10k_ft_efficientdet_d3_v2": "projects.EfficientDet.efficientdet",
            "seadronessee10k_ft_vitdet_b": "projects.ViTDet.vitdet",
            "waid10k_ft_vitdet_b": "projects.ViTDet.vitdet",
        }
        for model, module in expected.items():
            self.assertIn(module, self.records[model]["custom_imports"])


class ExporterTest(unittest.TestCase):
    def test_literal_only_changes_and_provenance_redaction(self):
        source = """# The historical training recipe stays unchanged.
data_root = '/private/storage/domain/'
load_from = '/private/storage/weight/model.pth'
work_dir = '/private/project/work/old'
lego_provenance = dict(source_config='/private/source/config.py',
                       load_from='/private/storage/weight/model.pth')
model = dict(loss=dict(type='FocalLoss', gamma=2.0))
train_dataloader = dict(batch_size=8)
"""
        result = EXPORT.sanitize_source(source, EXPORT.FORMAL_JOBS[0][0], "/private/storage")
        self.assertIn("data_root = 'data/domain/'", result)
        self.assertIn("load_from = 'checkpoints/model.pth'", result)
        self.assertIn("source_config='config.py'", result)
        self.assertIn("load_from='model.pth'", result)
        self.assertIn("model = dict(loss=dict(type='FocalLoss', gamma=2.0))", result)
        self.assertIn("train_dataloader = dict(batch_size=8)", result)
        self.assertIn("# The historical training recipe stays unchanged.", result)

    def test_rejects_unknown_absolute_path(self):
        with self.assertRaises(ValueError):
            EXPORT.sanitize_source("x = '/unknown/storage/file.json'\n",
                                   EXPORT.FORMAL_JOBS[0][0], "/private/storage")

    def test_rejects_private_path_in_comment(self):
        private_comment = "# /" + "home" + "/private/file\nx = 1\n"
        with self.assertRaises(ValueError):
            EXPORT.sanitize_source(private_comment,
                                   EXPORT.FORMAL_JOBS[0][0], "/private/storage")

    def test_rejects_traversal_and_unknown_model(self):
        with self.assertRaises(ValueError):
            EXPORT.sanitize_source("x = '/private/storage/../secret'\n",
                                   EXPORT.FORMAL_JOBS[0][0], "/private/storage")
        with self.assertRaises(ValueError):
            EXPORT.sanitize_source("x = 1\n", "unreviewed_model", "/private/storage")

    def test_queue_verification(self):
        jobs = " ".join(model + ":" + split for model, split in EXPORT.FORMAL_JOBS)
        EXPORT.verify_queue('JOBS=${JOBS:-"' + jobs + '"}\n')
        with self.assertRaises(ValueError):
            EXPORT.verify_queue('JOBS=${JOBS:-"other:val"}\n')


if __name__ == "__main__":
    unittest.main()
