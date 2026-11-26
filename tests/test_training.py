"""Shared contracts must remain usable without PyTorch or an HTTP framework."""

import dataclasses
import subprocess
import sys
import unittest

from app.core.training import EpochRecord, SurrogateTrainer, TrainingProgress, TrainingResult


class DummyTrainer(SurrogateTrainer):
    def load_dataset(self, data):
        pass

    def train(self):
        pass

    def predict(self, input_features):
        pass

    def export_checkpoint(self):
        return {}

    @classmethod
    def from_checkpoint(cls, checkpoint):
        return cls()


class TrainingContractTests(unittest.TestCase):
    def test_abstract_contract_includes_checkpoint_methods(self):
        self.assertEqual(SurrogateTrainer.__abstractmethods__,
                         {"load_dataset", "train", "predict", "export_checkpoint", "from_checkpoint"})
        with self.assertRaises(TypeError):
            SurrogateTrainer()

    def test_snapshots_are_immutable_and_instance_local(self):
        first, second = DummyTrainer(), DummyTrainer()
        initial = first.get_progress()
        self.assertEqual(initial, TrainingProgress())
        self.assertIsNone(initial.training_loss)
        self.assertIsNone(initial.validation_loss)
        self.assertIsNone(initial.best_validation_loss)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            initial.completed_steps = 12
        history = first.get_history()
        with first._lock:
            first._history.append(EpochRecord(1, 3, 4))
            first._progress = TrainingProgress(status="running", completed_steps=1)
        self.assertEqual(history, ())
        self.assertEqual(first.get_history(), (EpochRecord(1, 3, 4),))
        self.assertEqual(second.get_history(), ())
        self.assertEqual(second.get_progress(), initial)
        result = TrainingResult(first.get_history(), 1, 8, 2, 42)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.history[0].epoch = 9

    def test_import_and_subclass_without_backend_or_http(self):
        script = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'fastapi', 'starlette', 'httpx'}:
            raise AssertionError('Forbidden shared-interface dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from app.core.training import SurrogateTrainer
class Other(SurrogateTrainer):
    def load_dataset(self, data): pass
    def train(self): pass
    def predict(self, data): pass
    def export_checkpoint(self): return {}
    @classmethod
    def from_checkpoint(cls, data): return cls()
assert Other().get_history() == ()
'''
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
