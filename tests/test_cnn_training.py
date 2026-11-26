"""Behavioral tests use actual CPU PyTorch optimization, never mock training."""

import copy
import io
import math
import threading
import unittest
import warnings
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import numpy as np
import torch
import xarray as xr

from app.core.cnn_training import CNNTrainer
from app.core.data import TrainingData
from app.services.data_service import DataService


def dataset(samples=40, bins=5):
    x = np.random.default_rng(12).uniform(-1, 1, (samples, 2)).astype(np.float32)
    positions = np.linspace(-0.5, 0.5, bins)
    y = 1.5 * x[:, :1] - x[:, 1:] + positions - 1
    data = DataService().prepare_upload(x, y, ["temperature", "density"]).training_data
    data.target = data.target.assign_coords(bin=np.linspace(10, 20, bins))
    return data


class CNNTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def trainer(self, data=None, **settings):
        options = {"epochs": 4, "hidden_size": 8, "batch_size": 16}
        options.update(settings)
        trainer = CNNTrainer(**options)
        trainer.load_dataset(data if data is not None else dataset())
        return trainer

    def test_learning_best_weights_and_labelled_predictions(self):
        data = dataset(samples=64)
        trainer = self.trainer(data, epochs=100, learning_rate=0.01)
        initial_model = trainer._new_model(2, 5)
        initial_weights = {key: value.clone() for key, value in initial_model.state_dict().items()}
        result = trainer.train()
        self.assertEqual(len(result.history), 100)
        self.assertEqual(result.seed, 42)
        self.assertEqual((result.training_samples, result.validation_samples), (51, 13))
        self.assertLess(result.history[-1].training_loss, result.history[0].training_loss * 0.1)
        best = min(result.history, key=lambda record: record.validation_loss)
        self.assertLess(best.validation_loss, result.history[0].validation_loss * 0.1)
        self.assertEqual(result.best_epoch, best.epoch)
        self.assertTrue(all(math.isfinite(r.training_loss) and math.isfinite(r.validation_loss)
                            for r in result.history))
        weights = trainer.export_checkpoint()["weights"]
        self.assertTrue(any(not torch.equal(weights[key], value) for key, value in initial_weights.items()))
        prediction_input = data.input_features[:3].assign_coords(sample=["a", "c", "b"])
        prediction = trainer.predict(prediction_input)
        self.assertEqual(prediction.dims, ("sample", "bin"))
        self.assertEqual(prediction.shape, (3, 5))
        np.testing.assert_array_equal(prediction.sample, prediction_input.sample)
        np.testing.assert_array_equal(prediction.bin, data.target.bin)
        self.assertLess(float(prediction.min()), 0)  # unrestricted regression output
        _, y, _, valid = trainer._data
        all_predictions = trainer.predict(data.input_features).values
        self.assertAlmostEqual(float(np.mean((all_predictions[valid] - y[valid]) ** 2)),
                               best.validation_loss, places=6)
        self.assertEqual(trainer.get_progress().status, "completed")
        self.assertEqual(trainer.get_progress().best_validation_loss, best.validation_loss)
        self.assertEqual(trainer.get_history(), result.history)

    def test_best_epoch_restoration_when_last_epoch_is_worse(self):
        data = dataset()
        trainer = self.trainer(data, epochs=30, learning_rate=0.1)
        result = trainer.train()
        self.assertLess(result.best_epoch, trainer.config.epochs)
        validation = trainer._data[3]
        loss = np.mean((trainer.predict(data.input_features).values[validation] -
                        data.target.values[validation]) ** 2)
        self.assertAlmostEqual(float(loss), min(r.validation_loss for r in result.history), places=5)
        best_loss = min(r.validation_loss for r in result.history)
        self.assertGreater(result.history[-1].validation_loss, best_loss)
        self.assertEqual(trainer.get_progress().best_validation_loss, best_loss)

    def test_fixed_disjoint_split_training_only_scaling_and_snapshot(self):
        data = dataset(samples=11)
        data.input_features.values[:, 1] = 7
        original = data.input_features.values.copy()
        trainer = self.trainer(data, validation_fraction=0.21)
        x, y, train, valid = trainer._data
        self.assertEqual(len(valid), 3)
        self.assertEqual(set(train) & set(valid), set())
        self.assertEqual(set(train) | set(valid), set(range(11)))
        np.testing.assert_allclose(trainer._mean, original[train].astype(np.float64).mean(axis=0))
        self.assertEqual(trainer._scale[1], 1)
        np.testing.assert_allclose(x[train].mean(axis=0), 0, atol=1e-7)
        np.testing.assert_array_equal(y, data.target.values.astype(np.float32))
        data.input_features.values[:] = 999
        data.target.values[:] = 999
        self.assertFalse((x == 999).any())
        self.assertFalse((y == 999).any())
        trainer.train()
        trainer.train()
        np.testing.assert_array_equal(trainer._data[2], train)
        np.testing.assert_array_equal(trainer._data[3], valid)

    def test_small_split_bounds_and_single_bin(self):
        for fraction in (0.001, 0.999):
            with self.subTest(fraction=fraction):
                data = dataset(samples=2, bins=1)
                trainer = self.trainer(data, validation_fraction=fraction)
                result = trainer.train()
                self.assertEqual((result.training_samples, result.validation_samples), (1, 1))
                self.assertEqual(trainer.predict(data.input_features).shape, (2, 1))

    def test_seed_reproducibility_and_global_rng_isolation(self):
        np.random.seed(123)
        numpy_state = np.random.get_state()
        torch.manual_seed(123)
        torch_state = torch.get_rng_state().clone()
        data = dataset()
        first, second = self.trainer(data), self.trainer(data)
        a, b = first.train(), second.train()
        self.assertEqual(a, b)
        np.testing.assert_array_equal(first._data[2], second._data[2])
        np.testing.assert_array_equal(first.predict(data.input_features), second.predict(data.input_features))
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
        self.assertEqual(numpy_state[2:], np.random.get_state()[2:])
        different = self.trainer(data, seed=43)
        self.assertFalse(np.array_equal(first._data[2], different._data[2]))
        nondeterministic = self.trainer(data, seed=None)
        self.assertIsNone(nondeterministic.train().seed)

    def test_backend_uses_explicit_cpu_float32_not_global_defaults(self):
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            trainer = self.trainer()
            trainer.train()
            self.assertTrue(all(t.device.type == "cpu" and t.dtype == torch.float32
                                for t in trainer.export_checkpoint()["weights"].values()))
        finally:
            torch.set_default_dtype(old_dtype)

    def test_invalid_configuration(self):
        settings = [{"epochs": 0}, {"epochs": True}, {"batch_size": -1}, {"batch_size": 1.5},
                    {"hidden_size": 0}, {"validation_fraction": 0}, {"validation_fraction": 1},
                    {"validation_fraction": float("nan")}, {"validation_fraction": "0.2"},
                    {"learning_rate": 0}, {"learning_rate": float("inf")}, {"seed": -1},
                    {"seed": True}, {"seed": 2**63}]
        for setting in settings:
            with self.subTest(setting=setting), self.assertRaises((ValueError, TypeError)):
                CNNTrainer(**setting)

    def test_invalid_data(self):
        arrays = [(np.zeros((0, 2)), np.zeros((0, 3))),
                  (np.zeros((1, 2)), np.zeros((1, 3))),
                  (np.zeros((2, 0)), np.zeros((2, 3))),
                  (np.zeros((2, 2)), np.zeros((2, 0))),
                  (np.full((2, 2), np.nan), np.zeros((2, 3))),
                  (np.zeros((2, 2)), np.full((2, 3), np.inf)),
                  (np.full((2, 2), "x"), np.zeros((2, 3))),
                  (np.zeros((2, 2), dtype=complex), np.zeros((2, 3))),
                  (np.zeros((2, 2)), np.full((2, 3), 1e100))]
        for x, y in arrays:
            with self.subTest(shape=x.shape, dtype=x.dtype):
                trainer = CNNTrainer()
                with self.assertRaises((ValueError, TypeError)):
                    trainer.load_dataset(TrainingData(x, y, [str(i) for i in range(x.shape[1])]))
                with self.assertRaises(RuntimeError):
                    trainer.predict(dataset().input_features)
        with self.assertRaises(TypeError):
            CNNTrainer().load_dataset(object())
        data = dataset()
        data.target = data.target.assign_coords(bin=np.arange(5).astype("datetime64[D]"))
        with self.assertRaises(TypeError):
            CNNTrainer().load_dataset(data)

    def test_prediction_rejections(self):
        data = dataset()
        trainer = self.trainer(data)
        trainer.train()
        invalid = [data.input_features.isel(feature=[1, 0]), data.input_features.isel(feature=[0]),
                   data.input_features.transpose(), data.input_features.drop_vars("feature"),
                   xr.full_like(data.input_features, np.nan), xr.full_like(data.input_features, np.inf)]
        for values in invalid:
            with self.subTest(dims=values.dims), self.assertRaises(ValueError):
                trainer.predict(values)
        with self.assertRaises(TypeError):
            trainer.predict(data.input_features.values)
        with self.assertRaises(TypeError):
            trainer.predict(data.input_features.astype(str))
        self.assertEqual(trainer.predict(data.input_features[:0]).shape, (0, 5))

    def test_standardization_preserves_float64_feature_variation(self):
        data = dataset()
        data.input_features = data.input_features.astype(np.float64)
        data.input_features.values[:] += 1e10
        trainer = self.trainer(data)
        self.assertTrue((trainer._scale > 0.01).all())
        self.assertFalse((trainer._data[0] == 0).all())
        trainer.train()
        self.assertTrue(np.isfinite(trainer.predict(data.input_features)).all())

    def test_mutated_training_data_schema_is_revalidated(self):
        for change in (lambda d: setattr(d, "input_features", d.input_features.transpose()),
                       lambda d: setattr(d, "target", d.target[:2]),
                       lambda d: setattr(d, "input_features", d.input_features.drop_vars("feature")),
                       lambda d: setattr(d, "target", d.target.drop_vars("bin"))):
            data = dataset()
            change(data)
            with self.assertRaises(ValueError):
                CNNTrainer().load_dataset(data)

    def test_reordered_target_samples_are_rejected(self):
        data = dataset()
        data.target = data.target.isel(sample=slice(None, None, -1))
        with self.assertRaisesRegex(ValueError, "sample indexes must match in order"):
            CNNTrainer().load_dataset(data)

    def test_coordinates_must_label_their_named_dimensions(self):
        for attribute, name in (("input_features", "feature"), ("target", "bin")):
            for samples in (2 if name == "feature" else 5, 10):
                # Test even matching lengths: the dimension, not only size, matters.
                for scalar in (False, True):
                    with self.subTest(name=name, samples=samples, scalar=scalar):
                        data = dataset(samples=samples)
                        array = getattr(data, attribute).drop_vars(name)
                        coordinate = "label" if scalar else ("sample", np.arange(samples))
                        setattr(data, attribute, array.assign_coords({name: coordinate}))
                        with self.assertRaisesRegex(ValueError, "coordinate must label"):
                            CNNTrainer().load_dataset(data)

        data = dataset()
        trainer = self.trainer(data)
        trainer.train()
        for samples, scalar in ((10, False), (2, False), (2, True)):
            inputs = data.input_features[:samples].drop_vars("feature")
            # Matching labels and length must still be rejected on the wrong axis.
            coordinate = "temperature" if scalar else (
                "sample", np.resize(["temperature", "density"], samples))
            inputs = inputs.assign_coords(feature=coordinate)
            with self.subTest(samples=samples, scalar=scalar):
                with self.assertRaisesRegex(ValueError, "coordinate must label"):
                    trainer.predict(inputs)

    def test_nonfinite_feature_statistics_are_rejected_without_warnings(self):
        for values in (np.linspace(1e200, 2e200, 40), np.full(40, 1e308)):
            with self.subTest(value=values[0]):
                data = dataset()
                data.input_features = data.input_features.astype(np.float64)
                data.input_features.values[:, 0] = values
                trainer = CNNTrainer()
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    with self.assertRaisesRegex(ValueError, "mean and scale must be finite"):
                        trainer.load_dataset(data)
                self.assertIsNone(trainer._data)
                with self.assertRaises(RuntimeError):
                    trainer.export_checkpoint()

    def test_evaluation_batches_are_bounded_and_losses_are_sample_weighted(self):
        data = dataset(samples=23)
        trainer = self.trainer(data, epochs=1, batch_size=4, validation_fraction=0.35)
        model = trainer._new_model(2, 5)
        training_errors, validation_errors = [], []
        forwarded = []
        actual_mse = torch.nn.functional.mse_loss

        def record_forward(module, inputs, output):
            forwarded.append((module.training, len(inputs[0])))

        def record_mse(predicted, target, *args, **kwargs):
            errors = (predicted.detach().double() - target.detach().double()).square().numpy()
            records = training_errors if torch.is_grad_enabled() else validation_errors
            records.append(errors)
            return actual_mse(predicted, target, *args, **kwargs)

        handle = model.register_forward_hook(record_forward)
        try:
            with patch.object(trainer, "_new_model", return_value=model), patch.object(
                    torch.nn.functional, "mse_loss", side_effect=record_mse):
                result = trainer.train()
            self.assertEqual([len(e) for e in training_errors], [4, 4, 4, 2])
            self.assertEqual([len(e) for e in validation_errors], [4, 4, 1])
            self.assertTrue(all(size <= 4 for _, size in forwarded))
            self.assertEqual([size for training, size in forwarded if not training], [4, 4, 1])
            expected_training = float(np.concatenate(training_errors).mean())
            expected_validation = float(np.concatenate(validation_errors).mean())
            self.assertAlmostEqual(result.history[0].training_loss, expected_training, places=6)
            self.assertAlmostEqual(result.history[0].validation_loss, expected_validation, places=6)
            self.assertAlmostEqual(trainer.get_progress().training_loss, expected_training, places=6)
            self.assertAlmostEqual(trainer.get_progress().validation_loss, expected_validation, places=6)

            forwarded.clear()
            inputs = data.input_features.isel(sample=np.arange(22, -1, -1))
            prediction = trainer.predict(inputs)
            self.assertEqual(forwarded, [(False, 4)] * 5 + [(False, 3)])
            # Single-sample inference provides an independent ordering reference.
            expected = np.concatenate([trainer.predict(inputs[i:i + 1]).values
                                       for i in range(len(inputs))])
            np.testing.assert_allclose(prediction.values, expected, rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(prediction.sample, inputs.sample)
            np.testing.assert_array_equal(prediction.bin, data.target.bin)
        finally:
            handle.remove()

    def test_validation_is_gradient_free_and_does_not_update_weights(self):
        trainer = self.trainer(epochs=2)
        model = trainer._new_model(2, 5)
        validations = []
        weights_before = None

        def before_forward(module, inputs):
            nonlocal weights_before
            if not module.training:
                self.assertFalse(torch.is_grad_enabled())
                weights_before = {key: value.clone() for key, value in module.state_dict().items()}

        def after_forward(module, inputs, output):
            if not module.training:
                self.assertFalse(output.requires_grad)
                for key, value in module.state_dict().items():
                    self.assertTrue(torch.equal(value, weights_before[key]))
                validations.append(True)

        model.register_forward_pre_hook(before_forward)
        model.register_forward_hook(after_forward)
        # The instrumented model still performs real forward/backward/Adam work.
        with patch.object(trainer, "_new_model", return_value=model):
            trainer.train()
        self.assertEqual(len(validations), 2)

    def test_lifecycle_reset_and_training_failure_retains_last_epoch(self):
        data = dataset()
        trainer = CNNTrainer(epochs=4, hidden_size=8)
        self.assertEqual(trainer.get_progress().status, "not_started")
        self.assertIsNone(trainer.get_progress().training_loss)
        for action in (trainer.train, trainer.export_checkpoint, lambda: trainer.predict(data.input_features)):
            with self.assertRaises(RuntimeError):
                action()
        trainer.load_dataset(data)
        trainer.train()
        old_history = trainer.get_history()
        trainer.load_dataset(data)
        self.assertEqual(trainer.get_history(), ())
        self.assertEqual(trainer.get_progress().completed_steps, 0)
        self.assertEqual(len(old_history), 4)
        with self.assertRaises(RuntimeError):
            trainer.export_checkpoint()
        original_step = torch.optim.Adam.step
        calls = 0
        failure = ValueError("intentional optimizer failure")

        def fail_after_one_epoch(optimizer, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:  # 32 training samples, one batch per epoch
                raise failure
            return original_step(optimizer, *args, **kwargs)

        with patch.object(torch.optim.Adam, "step", fail_after_one_epoch):
            with self.assertRaises(ValueError) as error:
                trainer.train()
        self.assertIs(error.exception, failure)
        progress = trainer.get_progress()
        self.assertEqual(progress.status, "failed")
        self.assertEqual(progress.completed_steps, 1)
        self.assertEqual(len(trainer.get_history()), 1)
        self.assertEqual(progress.training_loss, trainer.get_history()[0].training_loss)
        self.assertIn("intentional", progress.message)
        with self.assertRaises(RuntimeError):
            trainer.predict(data.input_features)
        trainer.train()
        self.assertEqual(trainer.get_progress().status, "completed")

    def test_nonfinite_metrics_fail_clearly(self):
        data = dataset()
        data.target.values[:] = 1e30  # finite inputs; squared loss overflows float32
        trainer = self.trainer(data, learning_rate=np.float64(0.001))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            trainer.train()
        self.assertEqual(trainer.get_progress().status, "failed")
        self.assertEqual(trainer.get_progress().completed_steps, 0)
        with self.assertRaises(RuntimeError):
            trainer.export_checkpoint()

    def test_threaded_polling_coherence_and_overlap_guards(self):
        trainer = self.trainer(epochs=3, batch_size=64)
        data = dataset()
        entered, release = threading.Event(), threading.Event()
        original_step = torch.optim.Adam.step
        calls, failures = 0, []

        def pause_in_second_epoch(optimizer, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                entered.set()
                if not release.wait(10):
                    raise TimeoutError("Test did not release training")
            return original_step(optimizer, *args, **kwargs)

        def train():
            try:
                trainer.train()
            except BaseException as error:
                failures.append(error)

        with patch.object(torch.optim.Adam, "step", pause_in_second_epoch):
            thread = threading.Thread(target=train, daemon=True)
            thread.start()
            try:
                self.assertTrue(entered.wait(10))
                # Poll in another thread with a deadline: a whole-run lock must fail.
                snapshots = []
                poller = threading.Thread(target=lambda: snapshots.append(
                    (trainer.get_progress(), trainer.get_history())), daemon=True)
                poller.start()
                poller.join(2)
                self.assertFalse(poller.is_alive(), "Polling blocked behind training")
                progress, history = snapshots[0]
                self.assertEqual(progress.status, "running")
                self.assertEqual((progress.completed_steps, progress.total_steps, progress.step_unit), (1, 3, "epoch"))
                self.assertEqual(progress.training_loss, history[-1].training_loss)
                self.assertEqual(progress.validation_loss, history[-1].validation_loss)
                self.assertEqual(progress.best_validation_loss,
                                 min(record.validation_loss for record in history))
                with self.assertRaises(FrozenInstanceError):
                    history[0].epoch = 9
                for action in (trainer.train, lambda: trainer.load_dataset(data),
                               trainer.export_checkpoint, lambda: trainer.predict(data.input_features)):
                    with self.assertRaises(RuntimeError):
                        action()
            finally:
                release.set()
                thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(len(trainer.get_history()), 3)
        self.assertEqual(len(history), 1)
        self.assertEqual(progress.completed_steps, 1)
        self.assertEqual(trainer.get_progress().status, "completed")

    def test_checkpoint_weights_only_roundtrip_and_no_aliasing(self):
        data = dataset()
        # Object coordinates containing NumPy scalars must normalize safely.
        data.target = data.target.assign_coords(bin=np.array([np.int64(n) for n in range(5)], dtype=object))
        trainer = self.trainer(data, learning_rate=np.float64(0.001))
        trainer.train()
        expected = trainer.predict(data.input_features)
        checkpoint = trainer.export_checkpoint()
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        safe = torch.load(buffer, weights_only=True, map_location="cpu")
        restored = CNNTrainer.from_checkpoint(safe)
        self.assertIsNone(restored._data)
        xr.testing.assert_identical(expected, restored.predict(data.input_features))
        self.assertEqual(restored.config, trainer.config)
        self.assertEqual(restored.get_history(), ())
        for tensor in checkpoint["weights"].values():
            tensor.add_(100)
        checkpoint["mean"].add_(100)
        checkpoint["bins"]["values"][0] = -100
        for tensor in safe["weights"].values():
            tensor.zero_()
        safe["scale"].fill_(999)
        safe["features"]["values"][0] = "changed"
        xr.testing.assert_identical(expected, trainer.predict(data.input_features))
        xr.testing.assert_identical(expected, restored.predict(data.input_features))
        again = CNNTrainer.from_checkpoint(restored.export_checkpoint())
        xr.testing.assert_identical(expected, again.predict(data.input_features))

    def test_reject_invalid_or_incompatible_checkpoints(self):
        trainer = self.trainer()
        trainer.train()
        valid = trainer.export_checkpoint()
        corruptions = [lambda c: c.update(model_type="other"), lambda c: c.update(format_version=2),
                       lambda c: c.update(format_version=True), lambda c: c.pop("mean"),
                       lambda c: c.update(n_bins=0), lambda c: c["config"].update(epochs=0),
                       lambda c: c["config"].pop("seed"), lambda c: c["features"].update(values=[]),
                       lambda c: c["bins"].update(dtype="datetime64[D]"),
                       lambda c: c["scale"].fill_(0), lambda c: c["mean"].fill_(float("nan")),
                       lambda c: c.update(mean=torch.zeros(1)), lambda c: c["weights"].pop(next(iter(c["weights"]))),
                       lambda c: c["weights"][next(iter(c["weights"]))].fill_(float("inf")),
                       lambda c: c["weights"].update(encoder_bad=torch.zeros(1))]
        for corrupt in corruptions:
            checkpoint = copy.deepcopy(valid)
            corrupt(checkpoint)
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                CNNTrainer.from_checkpoint(checkpoint)
        for checkpoint in (None, {}, {"model_type": "CNNTrainer"}):
            with self.assertRaises(ValueError):
                CNNTrainer.from_checkpoint(checkpoint)

    def test_inconsistent_checkpoint_weights_rejected_before_model_allocation(self):
        trainer = self.trainer()
        trainer.train()
        valid = trainer.export_checkpoint()
        changes = [lambda c: c["config"].update(hidden_size=10**12),
                   lambda c: c["weights"].pop("encoder.0.bias"),
                   lambda c: c["weights"].update({"encoder.0.bias": torch.zeros(1)}),
                   lambda c: c["weights"].update({"encoder.0.bias": c["weights"]["encoder.0.bias"].double()}),
                   lambda c: c["weights"]["encoder.0.bias"].fill_(float("nan"))]
        for change in changes:
            checkpoint = copy.deepcopy(valid)
            change(checkpoint)
            with self.subTest(change=change), patch.object(CNNTrainer, "_new_model") as allocate:
                with self.assertRaisesRegex(ValueError, "model weight"):
                    CNNTrainer.from_checkpoint(checkpoint)
                allocate.assert_not_called()
