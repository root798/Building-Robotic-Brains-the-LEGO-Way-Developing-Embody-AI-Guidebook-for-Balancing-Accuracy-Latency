"""CPU-only tests for safe output handling; no deep-learning imports required."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from training.train import add_ssd_loss_compatibility, reserve_work_dir


class TrainingEntrypointTest(unittest.TestCase):
    def test_reserves_new_directory_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new-run"
            self.assertEqual(reserve_work_dir(output), output)
            self.assertTrue(output.is_dir())
            with self.assertRaises(FileExistsError):
                reserve_work_dir(output)

    def test_refuses_dangling_symlink_without_following_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "old-run-link"
            target = Path(temporary) / "missing-target"
            output.symlink_to(target, target_is_directory=True)
            with self.assertRaises(FileExistsError):
                reserve_work_dir(output)
            self.assertFalse(target.exists())

    def test_ssd_shim_preserves_existing_loss_and_sigmoid_choice(self):
        head = SimpleNamespace(use_sigmoid_cls=True)
        add_ssd_loss_compatibility(SimpleNamespace(bbox_head=head))
        self.assertTrue(head.loss_cls.use_sigmoid)
        original_loss = head.loss_cls
        add_ssd_loss_compatibility(SimpleNamespace(bbox_head=head))
        self.assertIs(head.loss_cls, original_loss)
        add_ssd_loss_compatibility(SimpleNamespace())


if __name__ == "__main__":
    unittest.main(verbosity=2)
