"""Strict transport contracts for training submission and progress."""

from typing import Annotated, Literal
from sys import float_info
from uuid import UUID

from pydantic import BeforeValidator, BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr

from app.core.cnn_training import CNNConfig


def finite_integer(value):
    if isinstance(value, int) and not -float_info.max <= value <= float_info.max:
        raise ValueError("Integer magnitude exceeds finite float64 range")
    return value


Number = Annotated[StrictInt | StrictFloat, Field(allow_inf_nan=False), BeforeValidator(finite_integer)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: PositiveInt = 100
    batch_size: PositiveInt = 32
    learning_rate: Annotated[Number, Field(gt=0)] = 0.001
    hidden_size: PositiveInt = 32
    validation_fraction: Annotated[Number, Field(gt=0, lt=1)] = 0.2
    seed: Annotated[StrictInt, Field(ge=0, lt=2**63)] | None = 42

    def to_core(self) -> CNNConfig:
        return CNNConfig(**self.model_dump())


class TrainingSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_features: list[list[Number]]
    target: list[list[Number]]
    input_feature_labels: list[StrictStr]
    config: TrainingConfig = Field(default_factory=TrainingConfig)


class TrainingRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid", strict=True)

    model_id: UUID
    dataset_id: UUID
    status: Literal["queued", "running", "saving", "completed", "failed"]
    completed_steps: Annotated[int, Field(ge=0)]
    total_steps: Annotated[int, Field(ge=0)]
    step_unit: str
    training_loss: Number | None
    validation_loss: Number | None
    best_validation_loss: Number | None
    message: str | None
    run_url: str | None
    artifact_reference: str | None


class TrainingErrorResponse(BaseModel):
    detail: str
