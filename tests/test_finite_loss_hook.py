"""CPU regression tests exercising the real MMEngine BaseModel.train_step.

Only the detector, optimizer actions and certificate writes are mocked. These
tests do not allocate CUDA tensors or write training artifacts. Run with
CUDA_VISIBLE_DEVICES="" to keep physical GPUs unavailable to the test process.
"""

import contextlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from mmengine.model import BaseModel

from training.reppoints_numerics import FiniteTrainingLossHook


class IdentityPreprocessor(torch.nn.Module):
    def forward(self, data, training=False):
        return data


class MinimalDetector(BaseModel):
    def __init__(self):
        super().__init__(data_preprocessor=IdentityPreprocessor())
        self.loss_value = 1.0

    def loss(self, inputs, data_samples):
        return {
            "loss_cls": [torch.tensor(0.5), torch.tensor(self.loss_value)],
            "loss_pts_init": [torch.tensor(2.0)],
        }

    def forward(self, inputs, data_samples=None, mode="tensor"):
        if mode != "loss":
            raise AssertionError("Expected BaseModel.train_step loss dispatch")
        return self.loss(inputs, data_samples)


class OptimizerSpy:
    def __init__(self):
        self.updated_losses = []

    def optim_context(self, model):
        return contextlib.nullcontext()

    def update_params(self, loss):
        self.updated_losses.append(float(loss))


class FiniteTrainingLossHookTest(unittest.TestCase):
    def setUp(self):
        self.writes = []

        def capture_write(path, text, *args, **kwargs):
            self.writes.append((path.name, json.loads(text)))
            return len(text)

        write_patch = patch("pathlib.Path.write_text", new=capture_write)
        write_patch.start()
        self.addCleanup(write_patch.stop)
        self.model = MinimalDetector()
        self.optimizer = OptimizerSpy()
        self.runner = SimpleNamespace(
            model=self.model, epoch=0, iter=0, max_iters=2,
            work_dir="reppoints-hook-artifacts-not-written",
        )
        self.hook = FiniteTrainingLossHook()
        self.data = {
            "inputs": torch.tensor(0.0),
            "data_samples": [SimpleNamespace(metainfo={"img_id": 7})],
        }

    def train_one(self):
        result = self.model.train_step(self.data, self.optimizer)
        self.runner.iter += 1
        return result

    def test_finite_batch_reaches_update_and_is_counted(self):
        self.hook.before_train(self.runner)
        result = self.train_one()
        self.assertEqual(self.hook.checked_batches, 1)
        self.assertEqual(self.optimizer.updated_losses, [3.5])
        self.assertEqual(float(result["loss"]), 3.5)
        self.assertFalse(self.writes)

    def test_nan_stops_before_update_and_records_component_and_image(self):
        self.hook.before_train(self.runner)
        self.train_one()
        self.model.loss_value = float("nan")
        with self.assertRaises(FloatingPointError):
            self.train_one()
        self.assertEqual(self.optimizer.updated_losses, [3.5])
        self.assertEqual(self.hook.checked_batches, 1)
        self.assertEqual(self.runner.iter, 1)
        self.assertEqual(self.writes, [("NONFINITE_LOSS.json", {
            "epoch": 1, "iteration": 2,
            "components": ["losses.loss_cls[1]"], "image_ids": [7],
        })])

    def test_early_stop_does_not_issue_completion_certificate(self):
        self.hook.before_train(self.runner)
        self.train_one()
        with self.assertRaisesRegex(RuntimeError, "expected batches"):
            self.hook.after_train(self.runner)
        self.assertEqual(self.model.loss, self.hook.original_loss)
        self.assertFalse(self.writes)

    def test_complete_run_issues_certificate_and_restores_loss_method(self):
        self.hook.before_train(self.runner)
        self.train_one()
        self.train_one()
        self.runner.epoch = 12
        self.hook.after_train(self.runner)
        self.assertEqual(self.model.loss, self.hook.original_loss)
        self.assertEqual(self.writes, [("FINITE_TRAINING.json", {
            "checked_batches": 2, "nonfinite_loss_batches": 0,
            "completed": True, "start_iteration": 0,
            "final_iteration": 2, "final_epoch": 12,
        })])

    def test_unchecked_iteration_does_not_issue_completion_certificate(self):
        self.hook.before_train(self.runner)
        self.train_one()
        self.runner.iter = self.runner.max_iters
        with self.assertRaisesRegex(RuntimeError, "expected batches"):
            self.hook.after_train(self.runner)
        self.assertFalse(self.writes)

    def test_resumed_run_counts_only_new_iterations(self):
        self.runner.iter = 10
        self.runner.max_iters = 12
        self.hook.before_train(self.runner)
        self.train_one()
        self.train_one()
        self.hook.after_train(self.runner)
        self.assertEqual(len(self.writes), 1)
        name, certificate = self.writes[0]
        self.assertEqual(name, "FINITE_TRAINING.json")
        self.assertEqual(certificate["start_iteration"], 10)
        self.assertEqual(certificate["checked_batches"], 2)
        self.assertEqual(certificate["final_iteration"], 12)
        self.assertTrue(certificate["completed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
