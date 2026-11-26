"""Process-local background CPU training with W&B history and checkpoints.

Keep one shared instance for live polling. Historical lookup is explicitly
network-bound; it does not recreate a trainer or resume optimization.
"""

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Literal
from uuid import UUID, uuid4

import torch
import wandb
from numpy.typing import NDArray

from app.core.cnn_training import CNNConfig, CNNTrainer
from app.services.data_service import DataService, PreparedDataset


@dataclass(frozen=True)
class TrainingRun:
    model_id: UUID
    dataset_id: UUID
    status: Literal["queued", "running", "saving", "completed", "failed"] = "queued"
    completed_steps: int = 0
    total_steps: int = 0
    step_unit: str = "epoch"
    training_loss: float | None = None
    validation_loss: float | None = None
    best_validation_loss: float | None = None
    message: str | None = None
    run_url: str | None = None
    artifact_reference: str | None = None


class TrainingRunNotFoundError(LookupError):
    """No local or historical run exists for the requested model ID."""


class TrainingService:
    """Queue submissions, tracking one training job at a time.

    Construction and submission perform no W&B calls. ``get_progress`` is
    local-only; use ``get_historical_progress`` for a network lookup. Call
    ``shutdown`` (or use a context manager) to drain accepted submissions.
    """

    def __init__(self, *, project: str, entity: str | None = None,
                 poll_interval: float = 0.1) -> None:
        if not isinstance(project, str) or not project.strip():
            raise ValueError("project must be a nonempty W&B project name")
        if entity is not None and (not isinstance(entity, str) or not entity.strip()):
            raise ValueError("entity must be a nonempty W&B entity name")
        if not isinstance(poll_interval, (int, float)) or not 0 < poll_interval < float("inf"):
            raise ValueError("poll_interval must be finite and positive")
        self.project, self.entity = project, entity
        self._poll_interval = poll_interval
        self._lock = Lock()
        self._runs: dict[UUID, TrainingRun] = {}
        self._closed = False
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="training-service")

    def submit(self, input_features: NDArray, target: NDArray,
               input_feature_labels: list[str], config: CNNConfig | None = None) -> TrainingRun:
        """Validate and snapshot before accepting; return IDs without waiting for training."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Training service is shut down")
        prepared = DataService().prepare_upload(input_features, target, input_feature_labels)
        return self.submit_prepared(prepared, config)

    def submit_prepared(self, prepared: PreparedDataset,
                        config: CNNConfig | None = None) -> TrainingRun:
        """Snapshot prepared data and preserve its ID before queue acceptance."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Training service is shut down")
        if not isinstance(prepared, PreparedDataset):
            raise TypeError("prepared must be PreparedDataset")
        if config is None:
            config = CNNConfig()
        if not isinstance(config, CNNConfig):
            raise TypeError("config must be CNNConfig")
        trainer = CNNTrainer(**asdict(config))
        trainer.load_dataset(prepared.training_data)
        snapshot = TrainingRun(uuid4(), prepared.dataset_id, total_steps=config.epochs)
        with self._lock:
            if self._closed:
                raise RuntimeError("Training service is shut down")
            self._runs[snapshot.model_id] = snapshot
            try:
                self._worker.submit(self._execute, snapshot.model_id, trainer)
            except BaseException:
                del self._runs[snapshot.model_id]
                raise
        return snapshot

    def get_progress(self, model_id: UUID | str) -> TrainingRun:
        """Return an immutable local snapshot without training or network waits."""
        model_id = UUID(str(model_id))
        with self._lock:
            try:
                return self._runs[model_id]
            except KeyError:
                raise TrainingRunNotFoundError(f"No local training run for {model_id}") from None

    def _update(self, model_id: UUID, **changes) -> TrainingRun:
        with self._lock:
            snapshot = replace(self._runs[model_id], **changes)
            self._runs[model_id] = snapshot
            return snapshot

    def _persist(self, run, snapshot: TrainingRun) -> None:
        payload = asdict(snapshot)
        payload.update(model_id=str(snapshot.model_id), dataset_id=str(snapshot.dataset_id))
        run.summary["training_service"] = payload

    def _execute(self, model_id: UUID, trainer: CNNTrainer) -> None:
        run = None
        logged = 0

        def drain(*, best_effort: bool = False) -> None:
            nonlocal logged
            # History and progress are separately locked in the core. Derive
            # completed metrics from one history snapshot to avoid torn reads.
            history = trainer.get_history()
            if history:
                last = history[-1]
                self._update(model_id, completed_steps=last.epoch,
                             training_loss=last.training_loss,
                             validation_loss=last.validation_loss,
                             best_validation_loss=min(r.validation_loss for r in history))
            if run is not None:
                while logged < len(history):
                    record = history[logged]
                    # An SDK error may occur after sending; never retry an
                    # ambiguous log call and potentially duplicate an epoch.
                    logged += 1
                    try:
                        run.log({"epoch": record.epoch, "training_loss": record.training_loss,
                                 "validation_loss": record.validation_loss}, step=record.epoch)
                    except BaseException:
                        if not best_effort:
                            raise
                self._persist(run, self.get_progress(model_id))

        # The executor context always joins optimization before failure cleanup
        # or checkpoint export, even when W&B monitoring raises mid-training.
        try:
            snapshot = self._update(model_id, status="running")
            run = wandb.init(project=self.project, entity=self.entity, id=model_id.hex,
                             resume="never", reinit="create_new",
                             config={"model_id": str(model_id),
                                     "dataset_id": str(snapshot.dataset_id),
                                     "model_type": "CNNTrainer", "cnn_config": asdict(trainer.config)})
            if run.disabled or run.offline:
                raise RuntimeError("Training service requires W&B online mode to upload model artifacts; "
                                   "enable W&B and set WANDB_MODE=online")
            self._update(model_id, run_url=run.url)
            self._persist(run, self.get_progress(model_id))
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="cnn-optimization") as training:
                future = training.submit(trainer.train)
                while not future.done():
                    drain()
                    wait([future], timeout=self._poll_interval)
                # Retrieve training errors first; failure cleanup drains history
                # without allowing a tracking error to replace that exception.
                result = future.result()
                drain()
            snapshot = self._update(model_id, status="saving")
            self._persist(run, snapshot)
            run.summary["training_result"] = {
                "best_epoch": result.best_epoch,
                "training_samples": result.training_samples,
                "validation_samples": result.validation_samples,
                "seed": result.seed,
            }
            with TemporaryDirectory(prefix="surrogate-checkpoint-") as directory:
                path = Path(directory) / "model.pt"
                torch.save(trainer.export_checkpoint(), path)
                artifact = wandb.Artifact(f"model-{model_id.hex}", type="model",
                                          metadata={"model_id": str(model_id),
                                                    "dataset_id": str(snapshot.dataset_id)})
                artifact.add_file(str(path))
                stored = run.log_artifact(artifact)
                stored.wait()
                reference = stored.qualified_name
                if not reference:
                    raise RuntimeError("W&B upload did not provide an artifact reference")
            snapshot = self._update(model_id, artifact_reference=reference)
            # Historical readers also require W&B's finished state, so an
            # interrupted/failed finalization cannot appear completed.
            self._persist(run, replace(snapshot, status="completed"))
            run.finish(exit_code=0)
            self._update(model_id, status="completed")
        except BaseException as exc:
            # A training executor, if started, has already been joined above.
            # Drain remaining completed epochs, preserving the original error.
            try:
                drain(best_effort=True)
            except BaseException:
                pass
            snapshot = replace(self.get_progress(model_id), status="failed",
                               message=str(exc) or type(exc).__name__)
            if run is not None:
                try:
                    self._persist(run, snapshot)
                except BaseException:
                    pass
                try:
                    run.finish(exit_code=1)
                except BaseException:
                    pass
            with self._lock:
                self._runs[model_id] = snapshot

    def get_historical_progress(self, model_id: UUID | str) -> TrainingRun:
        """Query W&B explicitly; missing runs differ from authentication/outage errors.

        Requires an explicit entity to identify a remote namespace reliably.
        Interrupted remote jobs are reported failed, never resumed.
        """
        model_id = UUID(str(model_id))
        if not self.entity:
            raise ValueError("Historical lookup requires a configured W&B entity")
        path = f"{self.entity}/{self.project}/{model_id.hex}"
        try:
            run = wandb.Api().run(path)
        except Exception as exc:
            # The SDK normalizes HTTPError/ValueError into CommError and keeps
            # the original exception in .exc (rather than .__cause__).
            original = getattr(exc, "exc", None)
            response = getattr(exc, "response", None)
            if response is None:
                response = getattr(original, "response", None)
            missing = getattr(response, "status_code", None) == 404
            missing |= isinstance(exc, (ValueError, wandb.errors.CommError)) and str(exc).lower().startswith(
                "could not find run ")
            if missing:
                raise TrainingRunNotFoundError(f"No W&B training run at {path}") from exc
            raise
        payload = dict(run.summary.get("training_service", {}))
        if not payload or payload.get("model_id") != str(model_id):
            raise ValueError(f"W&B run {path} has no matching training-service metadata")
        payload["model_id"] = model_id
        payload["dataset_id"] = UUID(payload["dataset_id"])
        payload["run_url"] = run.url
        snapshot = TrainingRun(**payload)
        # A historical query can reach a run still owned by another process.
        # W&B's running state alone is not evidence that training was interrupted.
        if run.state == "running" and snapshot.status in {"queued", "running", "saving"}:
            return snapshot
        if (run.state != "finished" or snapshot.status != "completed"
                or not snapshot.artifact_reference):
            snapshot = replace(snapshot, status="failed", message=snapshot.message or
                               f"Remote run did not complete successfully (W&B state: {run.state})")
        return snapshot

    def shutdown(self) -> None:
        """Reject new submissions and wait for every accepted run; safe to repeat."""
        with self._lock:
            self._closed = True
        self._worker.shutdown(wait=True)

    @property
    def is_shutdown(self) -> bool:
        """Whether draining has started and the service rejects new work."""
        with self._lock:
            return self._closed

    def __enter__(self) -> "TrainingService":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()
