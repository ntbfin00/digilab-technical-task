"""Service tests use controlled trainers and W&B doubles; no remote calls."""

from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
import time
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import torch
import wandb

from app.core.cnn_training import CNNConfig, CNNTrainer
from app.core.training import EpochRecord, TrainingProgress, TrainingResult
from app.services.data_service import DataService
from app.services.training_service import TrainingRunNotFoundError, TrainingService


class ControlledTrainer(CNNTrainer):
    """Real validation/copying, deterministic epochs and explicit thread barriers."""

    def __init__(self, **config):
        super().__init__(**config)
        self.started = Event()
        self.release = Event()
        self.ended = Event()
        self.error = None

    def train(self):
        self.started.set()
        try:
            if not self.release.wait(5):
                raise RuntimeError("test trainer release timed out")
            with self._lock:
                self._history = [EpochRecord(i, 1 / i, 2 / i)
                                 for i in range(1, self.config.epochs + 1)]
                last = self._history[-1]
                self._progress = TrainingProgress("completed", last.epoch, self.config.epochs,
                                                  "epoch", last.training_loss, last.validation_loss,
                                                  last.validation_loss)
            if self.error:
                raise self.error
            return TrainingResult(tuple(self._history), self.config.epochs, 3, 1, self.config.seed)
        finally:
            self.ended.set()

    def export_checkpoint(self):
        if not self.ended.is_set():
            raise AssertionError("export before training joined")
        return {"test": torch.tensor([1.0])}


class FakeArtifact:
    def __init__(self, name, type, metadata):
        self.name, self.type, self.metadata = name, type, metadata
        self.path = None
        self.checkpoint = None
        self.qualified_name = f"team/project/{name}:v0"
        self.waited = False

    def add_file(self, path):
        self.path = path

    def wait(self):
        # Read only on upload completion, exercising temporary file lifetime.
        self.checkpoint = torch.load(self.path, weights_only=True, map_location="cpu")
        self.waited = True
        return self


class FakeRun:
    def __init__(self):
        self.disabled = False
        self.offline = False
        self.url = "https://wandb.example/run"
        self.summary = {}
        self.logs = []
        self.finishes = []
        self.artifact = None
        self.state = "running"
        self.log_error = None
        self.finish_error = None
        self.upload_error = None
        self.log_called = Event()

    def log(self, values, step):
        self.logs.append((step, values))
        self.log_called.set()
        if self.log_error:
            raise self.log_error

    def log_artifact(self, artifact):
        if self.upload_error:
            raise self.upload_error
        self.artifact = artifact
        return artifact

    def finish(self, exit_code):
        self.finishes.append(exit_code)
        if self.finish_error:
            raise self.finish_error
        self.state = "finished" if exit_code == 0 else "failed"


class TrainingServiceTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(8, dtype=float).reshape(4, 2)
        self.y = np.arange(12, dtype=float).reshape(4, 3)
        self.config = CNNConfig(epochs=3, hidden_size=2)
        self.trainers = []
        self.runs = []
        self.services = []
        self.init_calls = []
        self.patchers = [
            patch("app.services.training_service.CNNTrainer", side_effect=self.make_trainer),
            patch("app.services.training_service.wandb.init", side_effect=self.make_run),
            patch("app.services.training_service.wandb.Artifact", FakeArtifact),
            patch("app.services.training_service.wandb.Api"),
        ]
        self.mocks = [p.start() for p in self.patchers]
        self.api = self.mocks[-1]

    def tearDown(self):
        for trainer in self.trainers:
            trainer.release.set()
        for service in self.services:
            service.shutdown()
        for patcher in reversed(self.patchers):
            patcher.stop()

    def make_trainer(self, **config):
        trainer = ControlledTrainer(**config)
        self.trainers.append(trainer)
        return trainer

    def make_run(self, **kwargs):
        self.init_calls.append(kwargs)
        run = FakeRun()
        self.runs.append(run)
        return run

    def service(self):
        service = TrainingService(project="project", entity="team", poll_interval=0.001)
        self.services.append(service)
        return service

    def submit(self, service):
        return service.submit(self.x, self.y, ["a", "b"], self.config)

    def terminal(self, service, model_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            snapshot = service.get_progress(model_id)
            if snapshot.status in {"completed", "failed"}:
                return snapshot
            time.sleep(0.001)
        self.fail(f"Run did not terminate: {service.get_progress(model_id)}")

    def test_submission_polling_queue_copy_and_shutdown(self):
        service = self.service()
        first = self.submit(service)
        self.assertTrue(self.trainers[0].started.wait(3))
        start = time.monotonic()
        running = service.get_progress(first.model_id)
        self.assertLess(time.monotonic() - start, 0.1)
        self.assertEqual(running.status, "running")
        with self.assertRaises(FrozenInstanceError):
            running.status = "failed"
        second = self.submit(service)
        self.assertEqual(service.get_progress(second.model_id).status, "queued")
        self.assertFalse(self.trainers[1].started.is_set())
        self.assertEqual(len({first.model_id, first.dataset_id, second.model_id, second.dataset_id}), 4)
        copied = self.trainers[1]._data[1].copy()
        normalized = self.trainers[1]._data[0].copy()
        self.x[:] = -100
        self.y[:] = -200
        np.testing.assert_array_equal(self.trainers[1]._data[1], copied)
        np.testing.assert_array_equal(self.trainers[1]._data[0], normalized)
        stopped = Event()
        thread = Thread(target=lambda: (service.shutdown(), stopped.set()))
        thread.start()
        try:
            self.assertFalse(stopped.wait(0.03))
            with self.assertRaisesRegex(RuntimeError, "shut down"):
                self.submit(service)
            self.trainers[0].release.set()
            self.assertTrue(self.trainers[1].started.wait(3))
            self.trainers[1].release.set()
            self.assertTrue(stopped.wait(3))
        finally:
            for trainer in self.trainers:
                trainer.release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        service.shutdown()
        for submitted in (first, second):
            self.assertEqual(service.get_progress(submitted.model_id).status, "completed")
        self.assertEqual(first.status, "queued")

    def test_reject_invalid_before_scheduling_or_wandb(self):
        service = self.service()
        cases = [([1], self.y, ["a", "b"], self.config, TypeError),
                 (np.zeros((4, 1)), self.y, ["a", "b"], self.config, ValueError),
                 (np.full((4, 2), np.nan), self.y, ["a", "b"], self.config, ValueError),
                 (self.x, self.y.astype(str), ["a", "b"], self.config, TypeError),
                 (self.x, self.y, ["a", "b"], {}, TypeError),
                 (self.x[:1], self.y[:1], ["a", "b"], self.config, ValueError)]
        for x, y, labels, config, error in cases:
            with self.subTest(error=error, config=config):
                with self.assertRaises(error):
                    service.submit(x, y, labels, config)
        with self.assertRaises(ValueError):
            CNNConfig(epochs=0)
        self.mocks[1].assert_not_called()
        self.assertEqual(service._runs, {})

    def test_scheduling_failure_rolls_back_acceptance(self):
        service = self.service()
        with patch.object(service._worker, "submit", side_effect=RuntimeError("cannot schedule")):
            with self.assertRaisesRegex(RuntimeError, "cannot schedule"):
                self.submit(service)
        self.assertEqual(service._runs, {})
        self.mocks[1].assert_not_called()

    def test_prepared_submission_preserves_id_and_snapshots_before_accepting(self):
        service = self.service()
        prepared = DataService().prepare_upload(self.x, self.y, ["a", "b"])
        with patch("app.services.training_service.DataService.prepare_upload") as prepare:
            submitted = service.submit_prepared(prepared, self.config)
        prepare.assert_not_called()
        self.assertEqual(submitted.dataset_id, prepared.dataset_id)
        copied = self.trainers[0]._data[1].copy()
        prepared.training_data.target.values[:] = -100
        np.testing.assert_array_equal(self.trainers[0]._data[1], copied)
        self.trainers[0].release.set()
        service.shutdown()
        with self.assertRaisesRegex(RuntimeError, "shut down"):
            service.submit_prepared(prepared)

    def test_prepared_scheduling_failure_rolls_back(self):
        service = self.service()
        prepared = DataService().prepare_upload(self.x, self.y, ["a", "b"])
        with patch.object(service._worker, "submit", side_effect=RuntimeError("cannot schedule")):
            with self.assertRaises(RuntimeError):
                service.submit_prepared(prepared)
        self.assertEqual(service._runs, {})

    def test_submission_and_polling_do_not_wait_for_network_initialization(self):
        entered, release = Event(), Event()

        def blocked_init(**kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("initialization barrier timed out")
            return self.make_run(**kwargs)

        self.mocks[1].side_effect = blocked_init
        service = self.service()
        try:
            submitted = self.submit(service)
            self.assertTrue(entered.wait(3))
            self.assertEqual(service.get_progress(submitted.model_id).status, "running")
            self.assertFalse(self.trainers[0].started.is_set())
        finally:
            release.set()
            for trainer in self.trainers:
                trainer.release.set()
        self.assertEqual(self.terminal(service, submitted.model_id).status, "completed")

    def test_fast_epochs_final_drain_metadata_and_storage(self):
        service = self.service()
        submitted = self.submit(service)
        self.assertTrue(self.trainers[0].started.wait(3))
        self.trainers[0].release.set()
        snapshot = self.terminal(service, submitted.model_id)
        self.assertEqual(snapshot.status, "completed")
        self.assertEqual(snapshot.completed_steps, 3)
        self.assertEqual(snapshot.validation_loss, 2 / 3)
        run = self.runs[0]
        self.assertEqual(run.logs, [
            (i, {"epoch": i, "training_loss": 1 / i, "validation_loss": 2 / i})
            for i in range(1, 4)
        ])
        self.assertEqual(run.finishes, [0])
        self.assertTrue(run.artifact.waited)
        self.assertFalse(Path(run.artifact.path).exists())
        self.assertEqual(snapshot.artifact_reference, run.artifact.qualified_name)
        self.assertEqual(run.summary["training_service"]["status"], "completed")
        self.assertEqual(run.summary["training_result"], {
            "best_epoch": 3, "training_samples": 3, "validation_samples": 1,
            "seed": self.config.seed,
        })
        self.assertNotIn("history", run.summary["training_result"])
        self.assertEqual(self.init_calls[0]["id"], submitted.model_id.hex)
        self.assertEqual(self.init_calls[0]["config"]["dataset_id"], str(submitted.dataset_id))
        self.assertEqual(self.init_calls[0]["config"]["cnn_config"], asdict(self.config))

    def test_training_failure_retains_epochs_and_drains(self):
        service = self.service()
        submitted = self.submit(service)
        self.trainers[0].error = ValueError("optimizer broke")
        self.trainers[0].release.set()
        snapshot = self.terminal(service, submitted.model_id)
        self.assertEqual(snapshot.status, "failed")
        self.assertEqual(snapshot.message, "optimizer broke")
        self.assertEqual(snapshot.completed_steps, 3)
        self.assertEqual(snapshot.best_validation_loss, 2 / 3)
        self.assertEqual([step for step, _ in self.runs[0].logs], [1, 2, 3])
        self.assertEqual(self.runs[0].finishes, [1])
        self.assertIsNone(self.runs[0].artifact)

    def test_disabled_and_offline_runs_fail_before_training(self):
        for mode in ("disabled", "offline"):
            with self.subTest(mode=mode):
                def init(**kwargs):
                    run = self.make_run(**kwargs)
                    setattr(run, mode, True)
                    return run

                self.mocks[1].side_effect = init
                service = self.service()
                submitted = self.submit(service)
                snapshot = self.terminal(service, submitted.model_id)
                self.assertEqual(snapshot.status, "failed")
                self.assertIn("WANDB_MODE=online", snapshot.message)
                self.assertFalse(self.trainers[-1].started.is_set())
                self.assertEqual(snapshot.completed_steps, 0)
                self.assertIsNone(self.runs[-1].artifact)
                self.assertEqual(self.runs[-1].finishes, [1])

    def test_failed_status_is_published_only_after_cleanup(self):
        entered, release = Event(), Event()
        service = self.service()
        submitted = self.submit(service)
        trainer = self.trainers[0]
        self.assertTrue(trainer.started.wait(3))
        trainer.error = ValueError("optimizer broke")
        run = self.runs[0]
        original = run.finish

        def blocked_finish(exit_code):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("cleanup barrier timed out")
            original(exit_code)

        with patch.object(run, "finish", side_effect=blocked_finish):
            trainer.release.set()
            try:
                self.assertTrue(entered.wait(3))
                snapshot = service.get_progress(submitted.model_id)
                self.assertEqual(snapshot.status, "running")
                self.assertEqual(snapshot.completed_steps, 3)
                self.assertEqual(run.summary["training_service"]["status"], "failed")
                self.assertEqual(run.finishes, [])
            finally:
                release.set()
            snapshot = self.terminal(service, submitted.model_id)
        self.assertEqual(snapshot.status, "failed")
        self.assertEqual(snapshot.message, "optimizer broke")
        self.assertEqual(run.finishes, [1])
        self.assertEqual(run.state, "failed")

    def test_intermediate_progress_while_training_is_blocked(self):
        service = self.service()
        submitted = self.submit(service)
        trainer = self.trainers[0]
        self.assertTrue(trainer.started.wait(3))
        run = self.runs[0]
        try:
            with trainer._lock:
                trainer._history = [EpochRecord(1, 1.0, 2.0)]
            self.assertTrue(run.log_called.wait(3))
            snapshot = service.get_progress(submitted.model_id)
            self.assertFalse(trainer.ended.is_set())
            self.assertEqual(snapshot.status, "running")
            self.assertEqual(snapshot.completed_steps, 1)
            self.assertEqual(snapshot.total_steps, 3)
            self.assertEqual(snapshot.step_unit, "epoch")
            self.assertEqual(snapshot.training_loss, 1.0)
            self.assertEqual(snapshot.validation_loss, 2.0)
            self.assertEqual(snapshot.best_validation_loss, 2.0)
            self.assertEqual(run.logs, [(1, {
                "epoch": 1, "training_loss": 1.0, "validation_loss": 2.0,
            })])
        finally:
            trainer.release.set()
        self.assertEqual(self.terminal(service, submitted.model_id).status, "completed")

    def test_initialization_failure_then_next_job_runs(self):
        service = self.service()
        self.mocks[1].side_effect = RuntimeError("authentication failed")
        failed = self.submit(service)
        snapshot = self.terminal(service, failed.model_id)
        self.assertEqual(snapshot.message, "authentication failed")
        self.assertEqual(snapshot.status, "failed")
        self.assertFalse(self.trainers[0].started.is_set())
        self.mocks[1].side_effect = self.make_run
        next_run = self.submit(service)
        self.trainers[-1].release.set()
        self.assertEqual(self.terminal(service, next_run.model_id).status, "completed")

    def test_log_failure_awaits_training_and_preserves_original_error(self):
        service = self.service()
        submitted = self.submit(service)
        trainer = self.trainers[0]
        self.assertTrue(trainer.started.wait(3))
        run = self.runs[0]
        run.log_error = RuntimeError("logging failed")
        run.finish_error = RuntimeError("cleanup failed")
        with trainer._lock:
            trainer._history = [EpochRecord(1, 1.0, 2.0)]
        self.assertTrue(run.log_called.wait(3))
        self.assertFalse(trainer.ended.is_set())
        self.assertNotEqual(service.get_progress(submitted.model_id).status, "failed")
        trainer.release.set()
        snapshot = self.terminal(service, submitted.model_id)
        self.assertTrue(trainer.ended.is_set())
        self.assertEqual(snapshot.status, "failed")
        self.assertEqual(snapshot.message, "logging failed")
        self.assertEqual(snapshot.completed_steps, 3)
        self.assertEqual(run.finishes, [1])
        self.assertEqual([step for step, _ in run.logs], [1, 2, 3])

    def test_storage_and_finalization_failures_never_complete(self):
        for stage in ("export", "serialize", "upload", "wait", "finish", "summary"):
            with self.subTest(stage=stage):
                service = self.service()
                submitted = self.submit(service)
                trainer = self.trainers[-1]
                self.assertTrue(trainer.started.wait(3))
                run = self.runs[-1]
                error = RuntimeError(f"{stage} failed")
                target = None
                if stage == "export":
                    target = patch.object(trainer, "export_checkpoint", side_effect=error)
                elif stage == "serialize":
                    target = patch("app.services.training_service.torch.save", side_effect=error)
                elif stage == "upload":
                    run.upload_error = error
                elif stage == "wait":
                    target = patch.object(FakeArtifact, "wait", side_effect=error)
                elif stage == "finish":
                    run.finish_error = error
                else:
                    class BrokenSummary(dict):
                        def __setitem__(self, key, value):
                            raise error
                    run.summary = BrokenSummary()
                if target:
                    target.start()
                try:
                    trainer.release.set()
                    snapshot = self.terminal(service, submitted.model_id)
                    service.shutdown()
                finally:
                    if target:
                        target.stop()
                self.assertEqual(snapshot.status, "failed")
                self.assertEqual(snapshot.completed_steps, 3)
                self.assertEqual(snapshot.message, f"{stage} failed")
                self.assertEqual(run.finishes[-1], 1)

    def test_saving_is_visible_before_upload_finishes(self):
        entered, release = Event(), Event()
        original = FakeArtifact.wait

        def blocked(artifact):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("upload barrier timed out")
            return original(artifact)

        service = self.service()
        with patch.object(FakeArtifact, "wait", blocked):
            submitted = self.submit(service)
            self.trainers[0].release.set()
            try:
                self.assertTrue(entered.wait(3))
                snapshot = service.get_progress(submitted.model_id)
                self.assertEqual(snapshot.status, "saving")
                self.assertEqual(snapshot.completed_steps, 3)
            finally:
                release.set()
            self.assertEqual(self.terminal(service, submitted.model_id).status, "completed")

    def test_remote_lookup_reconstructs_and_rejects_interrupted_completion(self):
        service = self.service()
        submitted = self.submit(service)
        self.trainers[0].release.set()
        expected = self.terminal(service, submitted.model_id)
        remote = self.runs[0]
        restarted = self.service()
        self.api.return_value.run.return_value = remote
        self.assertEqual(restarted.get_historical_progress(submitted.model_id), expected)
        self.api.return_value.run.assert_called_with(f"team/project/{submitted.model_id.hex}")
        self.assertEqual(len(self.trainers), 1)
        remote.state = "running"
        remote.summary["training_service"]["status"] = "running"
        self.assertEqual(restarted.get_historical_progress(submitted.model_id).status, "running")
        remote.summary["training_service"]["status"] = "completed"
        for state in ("running", "crashed", "failed"):
            remote.state = state
            actual = restarted.get_historical_progress(submitted.model_id)
            self.assertEqual(actual.status, "failed")
            self.assertEqual(actual.completed_steps, 3)
        remote.state = "finished"
        self.assertIsNotNone(remote.summary["training_service"]["artifact_reference"])
        for status in ("saving", "running", "failed"):
            remote.summary["training_service"]["status"] = status
            actual = restarted.get_historical_progress(submitted.model_id)
            self.assertEqual(actual.status, "failed")
            self.assertEqual(actual.completed_steps, 3)
        remote.summary["training_service"]["status"] = "completed"
        remote.summary["training_service"]["artifact_reference"] = None
        self.assertEqual(restarted.get_historical_progress(submitted.model_id).status, "failed")
        with self.assertRaises(TrainingRunNotFoundError):
            restarted.get_progress(submitted.model_id)

    def test_missing_remote_run_is_distinct_from_backend_errors(self):
        service = self.service()
        api = self.api.return_value.run
        api.side_effect = ValueError("Could not find run missing")
        with self.assertRaises(TrainingRunNotFoundError):
            service.get_historical_progress(uuid4())
        for error in (RuntimeError("backend down"), PermissionError("unauthorized"),
                      ValueError("invalid project"), wandb.errors.CommError("backend down"),
                      wandb.errors.AuthenticationError("unauthorized")):
            api.side_effect = error
            with self.assertRaises(type(error)) as raised:
                service.get_historical_progress(uuid4())
            self.assertIs(raised.exception, error)
        missing = RuntimeError("404")
        missing.response = Mock(status_code=404)
        api.side_effect = missing
        with self.assertRaises(TrainingRunNotFoundError):
            service.get_historical_progress(uuid4())
        # Exercise the SDK's actual exception normalization without a network.
        from wandb.apis.normalize import normalize_exceptions

        @normalize_exceptions
        def absent(path):
            raise ValueError("Could not find run missing")

        api.side_effect = absent
        with self.assertRaises(TrainingRunNotFoundError):
            service.get_historical_progress(uuid4())
        api.side_effect = wandb.errors.CommError("HTTP 404", missing)
        with self.assertRaises(TrainingRunNotFoundError):
            service.get_historical_progress(uuid4())
        no_entity = TrainingService(project="project")
        try:
            with self.assertRaisesRegex(ValueError, "entity"):
                no_entity.get_historical_progress(uuid4())
        finally:
            no_entity.shutdown()

    def test_remote_failure_retains_original_message_and_metrics(self):
        service = self.service()
        submitted = self.submit(service)
        self.trainers[0].error = ValueError("bad optimization")
        self.trainers[0].release.set()
        expected = self.terminal(service, submitted.model_id)
        service.shutdown()
        self.api.return_value.run.return_value = self.runs[0]
        restarted = self.service()
        self.assertEqual(restarted.get_historical_progress(submitted.model_id), expected)
        self.assertEqual(expected.message, "bad optimization")
        self.assertEqual(expected.completed_steps, 3)

    def test_actual_cpu_checkpoint_predicts_identically(self):
        # Temporarily restore the real backend, preserving the mocked W&B SDK.
        captured = []

        def actual(**config):
            trainer = CNNTrainer(**config)
            captured.append(trainer)
            return trainer

        with patch("app.services.training_service.CNNTrainer", side_effect=actual):
            service = self.service()
            submitted = self.submit(service)
            snapshot = self.terminal(service, submitted.model_id)
        self.assertEqual(snapshot.status, "completed", snapshot.message)
        restored = CNNTrainer.from_checkpoint(self.runs[0].artifact.checkpoint)
        features = DataService().prepare_upload(self.x, self.y, ["a", "b"]).training_data.input_features
        np.testing.assert_array_equal(captured[0].predict(features).values,
                                      restored.predict(features).values)
        self.assertEqual([step for step, _ in self.runs[0].logs], [1, 2, 3])

    def test_import_and_use_without_http_framework(self):
        script = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".")[0] in {"fastapi", "starlette", "uvicorn"}:
        raise AssertionError("HTTP framework imported: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from app.services.training_service import TrainingService, TrainingRunNotFoundError
from uuid import uuid4
with TrainingService(project="test") as service:
    try:
        service.get_progress(uuid4())
    except TrainingRunNotFoundError:
        pass
'''
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
