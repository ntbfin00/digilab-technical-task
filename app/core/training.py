"""Framework-independent synchronous surrogate training contracts.

Snapshots are immutable and guarded by short-lived locks. They are local to this
process; services own background execution, persistence, and monitoring.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import RLock
from typing import Any, Literal, Mapping

import xarray as xr

from app.core.data import TrainingData


@dataclass(frozen=True)
class TrainingProgress:
    status: Literal["not_started", "running", "completed", "failed"] = "not_started"
    completed_steps: int = 0
    total_steps: int | None = None
    step_unit: str = "epoch"
    training_loss: float | None = None
    validation_loss: float | None = None
    best_validation_loss: float | None = None
    message: str | None = None


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    training_loss: float
    validation_loss: float


@dataclass(frozen=True)
class TrainingResult:
    history: tuple[EpochRecord, ...]
    best_epoch: int
    training_samples: int
    validation_samples: int
    seed: int | None


class SurrogateTrainer(ABC):
    """Implementations synchronize mutations with the same snapshot lock."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._progress = TrainingProgress()
        self._history: list[EpochRecord] = []

    def get_progress(self) -> TrainingProgress:
        with self._lock:
            return self._progress

    def get_history(self) -> tuple[EpochRecord, ...]:
        with self._lock:
            return tuple(self._history)

    @abstractmethod
    def load_dataset(self, data: TrainingData) -> None:
        """Snapshot new data and invalidate any previously fitted model."""

    @abstractmethod
    def train(self) -> TrainingResult:
        """Fit synchronously; reject concurrent mutation."""

    @abstractmethod
    def predict(self, input_features: xr.DataArray) -> xr.DataArray:
        """Predict using matching, ordered feature coordinates."""

    @abstractmethod
    def export_checkpoint(self) -> Mapping[str, Any]:
        """Return detached fitted state, without performing serialization or I/O."""

    @classmethod
    @abstractmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> "SurrogateTrainer":
        """Restore prediction without training data; reject incompatible state."""
