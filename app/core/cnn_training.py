"""CPU PyTorch dense-to-convolutional regression backend.

Explicit seeds reproduce runs on the same CPU/software environment. Local NumPy
and Torch generators isolate splitting and shuffling. Initialization saves and
restores Torch's CPU RNG under a backend-local lock; unrelated code using the
global RNG concurrently cannot be isolated. Cross-version/hardware bitwise
reproducibility is not guaranteed.
"""

import logging
import math
import secrets
from dataclasses import asdict, dataclass, replace
from threading import Lock
from typing import Any, Mapping

import numpy as np
import torch
import xarray as xr
from torch import nn

from app.core.data import TrainingData
from app.core.training import EpochRecord, SurrogateTrainer, TrainingProgress, TrainingResult

logger = logging.getLogger(__name__)
_INITIALIZATION_LOCK = Lock()


@dataclass(frozen=True)
class CNNConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    hidden_size: int = 32
    validation_fraction: float = 0.2
    seed: int | None = 42

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "hidden_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "validation_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be real numeric")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            # Keep checkpoint configuration safe for weights_only deserialization,
            # even if a caller supplies a NumPy floating-point scalar.
            object.__setattr__(self, name, float(value))
        if self.validation_fraction >= 1:
            raise ValueError("validation_fraction must be between zero and one")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**63):
            raise ValueError("seed must be None or an integer in [0, 2**63)")


class _Network(nn.Module):
    def __init__(self, features: int, bins: int, hidden: int) -> None:
        super().__init__()
        self.bins = bins
        self.hidden = hidden
        options = {"device": "cpu", "dtype": torch.float32}
        self.encoder = nn.Sequential(nn.Linear(features, hidden, **options), nn.ReLU(),
                                     nn.Linear(hidden, hidden * bins, **options), nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=1, **options),
                                     nn.ReLU(), nn.Conv1d(hidden, 1, 1, **options))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(values).reshape(-1, self.hidden, self.bins)).squeeze(1)


def _numeric(values: Any, name: str, dtype: Any = np.float32) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    with np.errstate(over="ignore"):
        result = array.astype(dtype, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} is outside {np.dtype(dtype).name} range")
    return result


def _encode_coordinate(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "biufUOS":
        raise TypeError("Unsupported coordinate type")
    encoded = []
    for value in values:
        value = value.item() if isinstance(value, np.generic) else value
        if type(value) not in (str, int, float, bool, bytes):
            raise TypeError("Coordinates must contain primitive scalar labels")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Coordinates must be finite")
        encoded.append(value)
    return {"dtype": values.dtype.str, "values": encoded}


def _dimension_coordinate(array: xr.DataArray, name: str) -> xr.DataArray:
    if name not in array.coords:
        raise ValueError(f"Missing {name} coordinate")
    coordinate = array.coords[name]
    if coordinate.dims != (name,) or coordinate.size != array.sizes[name]:
        raise ValueError(f"{name} coordinate must label the {name} dimension")
    return coordinate


def _decode_coordinate(value: Any, size: int) -> np.ndarray:
    if not isinstance(value, dict) or set(value) != {"dtype", "values"}:
        raise ValueError("Invalid coordinate metadata")
    if not isinstance(value["dtype"], str) or not isinstance(value["values"], list):
        raise ValueError("Invalid coordinate encoding")
    if len(value["values"]) != size:
        raise ValueError("Coordinate length mismatch")
    for scalar in value["values"]:
        if type(scalar) not in (str, int, float, bool, bytes):
            raise ValueError("Invalid coordinate scalar")
    result = np.array(value["values"], dtype=value["dtype"])
    if _encode_coordinate(result) != value:
        raise ValueError("Coordinate encoding is lossy")
    return result


class CNNTrainer(SurrogateTrainer):
    """Train unordered features into a locally correlated ordered target vector."""

    def __init__(self, *, epochs: int = 100, batch_size: int = 32,
                 learning_rate: float = 0.001, hidden_size: int = 32,
                 validation_fraction: float = 0.2, seed: int | None = 42) -> None:
        super().__init__()
        self._config = CNNConfig(epochs, batch_size, learning_rate, hidden_size,
                                 validation_fraction, seed)
        self._data = None
        self._model = None
        self._fitted = False
        self._busy = False
        self._progress = TrainingProgress(total_steps=epochs)

    @property
    def config(self) -> CNNConfig:
        return self._config

    def _claim(self) -> None:
        with self._lock:
            if self._busy:
                raise RuntimeError("Training or dataset replacement is already running")
            self._busy = True

    def load_dataset(self, data: TrainingData) -> None:
        self._claim()
        try:
            if not isinstance(data, TrainingData):
                raise TypeError("data must be TrainingData")
            if data.input_features.dims != ("sample", "feature") or data.target.dims != ("sample", "bin"):
                raise ValueError("Expected (sample, feature) inputs and (sample, bin) targets")
            features = _dimension_coordinate(data.input_features, "feature").values.copy()
            bins = _dimension_coordinate(data.target, "bin").values.copy()
            if not data.input_features.get_index("sample").equals(data.target.get_index("sample")):
                raise ValueError("Input and target sample indexes must match in order")
            x = _numeric(data.input_features.values, "input_features", np.float64)
            y = _numeric(data.target.values, "target")
            if x.shape[0] < 2 or x.shape[1] < 1 or y.shape[1] < 1 or x.shape[0] != y.shape[0]:
                raise ValueError("Require at least two samples and positive feature/bin counts")
            _encode_coordinate(features)
            _encode_coordinate(bins)
            permutation = np.random.default_rng(self.config.seed).permutation(len(x))
            count = min(len(x) - 1, max(1, math.ceil(len(x) * self.config.validation_fraction)))
            validation_indices, training_indices = permutation[:count], permutation[count:]
            # Float64 statistics avoid overflow/cancellation before float32 training.
            with np.errstate(over="ignore", invalid="ignore"):
                mean = x[training_indices].astype(np.float64).mean(axis=0)
                scale = x[training_indices].astype(np.float64).std(axis=0)
            if not np.isfinite(mean).all() or not np.isfinite(scale).all():
                raise ValueError("Feature mean and scale must be finite")
            scale[scale == 0] = 1
            normalized = _numeric((x.astype(np.float64) - mean) / scale, "standardized features")
            with self._lock:
                self._data = (normalized, y, training_indices, validation_indices)
                self._mean, self._scale = mean.copy(), scale.copy()
                self._features, self._bins = features, bins
                self._model = None
                self._fitted = False
                self._history = []
                self._progress = TrainingProgress(total_steps=self.config.epochs)
        finally:
            with self._lock:
                self._busy = False

    def _new_model(self, features: int, bins: int) -> _Network:
        with _INITIALIZATION_LOCK, torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(
                self.config.seed if self.config.seed is not None else secrets.randbits(63))
            return _Network(features, bins, self.config.hidden_size).cpu()

    def train(self) -> TrainingResult:
        self._claim()
        try:
            if self._data is None:
                raise RuntimeError("Load a dataset before training")
            with self._lock:
                self._fitted = False
                self._model = None
                self._history = []
                self._progress = TrainingProgress(status="running", total_steps=self.config.epochs)
            x, y, training_indices, validation_indices = self._data
            model = self._new_model(x.shape[1], y.shape[1])
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.config.seed if self.config.seed is not None else secrets.randbits(63))
            tx, ty = torch.from_numpy(x[training_indices]), torch.from_numpy(y[training_indices])
            vx, vy = torch.from_numpy(x[validation_indices]), torch.from_numpy(y[validation_indices])
            best_loss, best_epoch, best_weights = math.inf, 0, None
            for epoch in range(1, self.config.epochs + 1):
                model.train()
                order = torch.randperm(len(tx), generator=generator, device="cpu")
                total_loss = 0.0
                for indices in order.split(self.config.batch_size):
                    optimizer.zero_grad()
                    loss = nn.functional.mse_loss(model(tx[indices]), ty[indices])
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(indices)
                training_loss = total_loss / len(tx)
                model.eval()
                with torch.no_grad():
                    validation_total = 0.0
                    for start in range(0, len(vx), self.config.batch_size):
                        batch_x = vx[start:start + self.config.batch_size]
                        batch_y = vy[start:start + self.config.batch_size]
                        validation_total += nn.functional.mse_loss(model(batch_x), batch_y).item() * len(batch_x)
                    validation_loss = validation_total / len(vx)
                if not math.isfinite(training_loss) or not math.isfinite(validation_loss):
                    raise ValueError("Nonfinite training/validation metrics")
                if validation_loss < best_loss:
                    best_loss, best_epoch = validation_loss, epoch
                    best_weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
                with self._lock:
                    self._history.append(EpochRecord(epoch, training_loss, validation_loss))
                    self._progress = TrainingProgress("running", epoch, self.config.epochs, "epoch",
                                                      training_loss, validation_loss, best_loss)
                logger.info(
                    "Epoch %s/%s: training_loss=%g validation_loss=%g best_validation_loss=%g",
                    epoch, self.config.epochs, training_loss, validation_loss, best_loss,
                )
            model.load_state_dict(best_weights)
            model.eval()
            with self._lock:
                self._model, self._fitted = model, True
                self._progress = replace(self._progress, status="completed")
                result = TrainingResult(tuple(self._history), best_epoch, len(tx), len(vx), self.config.seed)
            logger.info("Surrogate training completed; best epoch %s", best_epoch)
            return result
        except BaseException as exc:
            with self._lock:
                self._fitted = False
                self._model = None
                self._progress = replace(self._progress, status="failed", message=str(exc))
            raise
        finally:
            with self._lock:
                self._busy = False

    def predict(self, input_features: xr.DataArray) -> xr.DataArray:
        with self._lock:
            if self._busy or not self._fitted:
                raise RuntimeError("Prediction requires a successfully fitted, non-running trainer")
            model = self._model
            mean, scale = self._mean, self._scale
            features, bins = self._features, self._bins.copy()
        if not isinstance(input_features, xr.DataArray):
            raise TypeError("input_features must be a labelled DataArray")
        if input_features.dims != ("sample", "feature") or "feature" not in input_features.coords:
            raise ValueError("Expected labelled (sample, feature) inputs")
        if not np.array_equal(_dimension_coordinate(input_features, "feature").values, features):
            raise ValueError("Feature coordinates must match training order")
        x = _numeric(input_features.values, "input_features", np.float64)
        normalized = _numeric((x.astype(np.float64) - mean) / scale, "standardized features")
        predictions = np.empty((len(x), len(bins)), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(x), self.config.batch_size):
                stop = start + self.config.batch_size
                predictions[start:stop] = model(torch.from_numpy(normalized[start:stop])).numpy()
        if not np.isfinite(predictions).all():
            raise ValueError("Nonfinite predictions")
        return xr.DataArray(predictions, dims=("sample", "bin"),
                            coords={"sample": input_features.coords["sample"].values.copy(), "bin": bins},
                            name="prediction")

    def export_checkpoint(self) -> dict[str, Any]:
        with self._lock:
            if self._busy or not self._fitted:
                raise RuntimeError("Checkpoint export requires a fitted, non-running trainer")
            model, mean, scale = self._model, self._mean, self._scale
            features, bins = self._features, self._bins
        return {"model_type": "CNNTrainer", "format_version": 1,
                "config": asdict(self.config), "n_features": len(features), "n_bins": len(bins),
                "features": _encode_coordinate(features), "bins": _encode_coordinate(bins),
                "mean": torch.tensor(mean.copy(), dtype=torch.float64, device="cpu"),
                "scale": torch.tensor(scale.copy(), dtype=torch.float64, device="cpu"),
                "weights": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> "CNNTrainer":
        try:
            if not isinstance(checkpoint, Mapping):
                raise ValueError("Checkpoint must be a mapping")
            required = {"model_type", "format_version", "config", "n_features", "n_bins",
                        "features", "bins", "mean", "scale", "weights"}
            if set(checkpoint) != required or checkpoint["model_type"] != "CNNTrainer":
                raise ValueError("Invalid checkpoint schema/model type")
            if type(checkpoint["format_version"]) is not int or checkpoint["format_version"] != 1:
                raise ValueError("Unsupported checkpoint format version")
            features, bins = checkpoint["n_features"], checkpoint["n_bins"]
            if any(type(n) is not int or n < 1 for n in (features, bins)):
                raise ValueError("Invalid architecture dimensions")
            config = checkpoint["config"]
            if not isinstance(config, dict) or set(config) != set(CNNConfig.__dataclass_fields__):
                raise ValueError("Invalid checkpoint configuration")
            trainer = cls(**config)
            trainer._features = _decode_coordinate(checkpoint["features"], features)
            trainer._bins = _decode_coordinate(checkpoint["bins"], bins)
            for name in ("mean", "scale"):
                tensor = checkpoint[name]
                if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.float64:
                    raise ValueError("Invalid feature scaling tensor")
                if tensor.shape != (features,) or not torch.isfinite(tensor).all():
                    raise ValueError("Invalid feature scaling values")
                if name == "scale" and not (tensor > 0).all():
                    raise ValueError("Feature scales must be positive")
                setattr(trainer, "_" + name, tensor.detach().numpy().copy())
            weights = checkpoint["weights"]
            hidden = trainer.config.hidden_size
            expected = {
                "encoder.0.weight": (hidden, features),
                "encoder.0.bias": (hidden,),
                "encoder.2.weight": (hidden * bins, hidden),
                "encoder.2.bias": (hidden * bins,),
                "decoder.0.weight": (hidden, hidden, 3),
                "decoder.0.bias": (hidden,),
                "decoder.2.weight": (1, hidden, 1),
                "decoder.2.bias": (1,),
            }
            if not isinstance(weights, dict) or set(weights) != set(expected):
                raise ValueError("Invalid model weight keys")
            for key, tensor in weights.items():
                if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
                    raise ValueError("Model weights must be CPU tensors")
                if tensor.shape != expected[key] or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                    raise ValueError("Invalid model weight shape, dtype, or values")
            model = trainer._new_model(features, bins)
            model.load_state_dict(weights, strict=True)
            model.eval()
            trainer._model, trainer._fitted = model, True
            trainer._progress = TrainingProgress(status="completed", total_steps=0,
                                                  message="Restored for prediction; no training history")
            return trainer
        except (TypeError, KeyError, RuntimeError, OverflowError) as exc:
            raise ValueError(f"Invalid checkpoint: {exc}") from exc
