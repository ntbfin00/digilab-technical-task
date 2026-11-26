"""Opt-in component test of the public core surrogate-training workflow.

Run explicitly from the repository root:
    .venv/bin/python -m pytest tests/component_surrogate_training.py -q

The filename intentionally avoids default pytest/unittest discovery patterns.
"""

import io
import math
import unittest

import numpy as np
import torch
import xarray as xr

from app.core.cnn_training import CNNTrainer
from app.core.data import TrainingData
from app.core.training import SurrogateTrainer, TrainingResult


class SurrogateTrainingComponentTests(unittest.TestCase):
    def test_training_prediction_and_checkpoint_round_trip(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)

        rng = np.random.default_rng(12)
        features = rng.uniform(-1, 1, (64, 2)).astype(np.float32)
        bin_offsets = np.linspace(-0.5, 0.5, 5)
        targets = 1.5 * features[:, :1] - features[:, 1:] + bin_offsets - 1
        data = TrainingData(features, targets, ["temperature", "density"])
        data.target = data.target.assign_coords(bin=np.linspace(10, 20, 5))

        trainer: SurrogateTrainer = CNNTrainer(
            epochs=100, batch_size=16, hidden_size=8,
            learning_rate=0.01, validation_fraction=0.2, seed=42,
        )
        self.assertEqual(trainer.get_progress().status, "not_started")
        trainer.load_dataset(data)
        result = trainer.train()

        self.assertIsInstance(result, TrainingResult)
        self.assertEqual((result.training_samples, result.validation_samples), (51, 13))
        self.assertEqual(result.seed, 42)
        self.assertEqual([record.epoch for record in result.history], list(range(1, 101)))
        self.assertTrue(all(
            math.isfinite(record.training_loss) and math.isfinite(record.validation_loss)
            for record in result.history
        ))
        best = min(result.history, key=lambda record: record.validation_loss)
        self.assertEqual(result.best_epoch, best.epoch)
        self.assertLess(result.history[-1].training_loss,
                        result.history[0].training_loss * 0.5)
        self.assertLess(best.validation_loss, result.history[0].validation_loss * 0.5)
        self.assertEqual(trainer.get_history(), result.history)
        progress = trainer.get_progress()
        self.assertEqual(progress.status, "completed")
        self.assertEqual((progress.completed_steps, progress.total_steps), (100, 100))
        self.assertEqual(progress.training_loss, result.history[-1].training_loss)
        self.assertEqual(progress.validation_loss, result.history[-1].validation_loss)
        self.assertEqual(progress.best_validation_loss, best.validation_loss)

        # New samples check useful inference rather than memorization of the dataset.
        new_features = rng.uniform(-1, 1, (12, 2)).astype(np.float32)
        expected = 1.5 * new_features[:, :1] - new_features[:, 1:] + bin_offsets - 1
        inputs = xr.DataArray(
            new_features, dims=("sample", "feature"),
            coords={"sample": [f"unseen-{i}" for i in range(12)],
                    "feature": ["temperature", "density"]},
        )
        prediction = trainer.predict(inputs)
        self.assertIsInstance(prediction, xr.DataArray)
        self.assertEqual(prediction.dims, ("sample", "bin"))
        self.assertEqual(prediction.shape, (12, 5))
        np.testing.assert_array_equal(prediction.sample, inputs.sample)
        np.testing.assert_array_equal(prediction.bin, data.target.bin)
        self.assertTrue(np.isfinite(prediction.values).all())
        baseline_mse = float(np.mean((expected - targets.mean(axis=0)) ** 2))
        prediction_mse = float(np.mean((prediction.values - expected) ** 2))
        self.assertLess(prediction_mse, baseline_mse * 0.25)

        with io.BytesIO() as buffer:
            torch.save(trainer.export_checkpoint(), buffer)
            buffer.seek(0)
            checkpoint = torch.load(buffer, weights_only=True, map_location="cpu")
        restored = CNNTrainer.from_checkpoint(checkpoint)
        xr.testing.assert_identical(prediction, restored.predict(inputs))


if __name__ == "__main__":
    unittest.main()
