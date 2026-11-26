"""Prepare labelled training datasets independently of any API framework."""

from dataclasses import dataclass
from uuid import UUID, uuid4

from numpy.typing import NDArray

from app.core.data import TrainingData


@dataclass
class PreparedDataset:
    """A dataset identifier and training data ready to pass to training code."""

    dataset_id: UUID
    training_data: TrainingData


class DataService:
    """Prepare datasets without storing them or retaining service-level state."""

    def prepare_upload(
        self,
        input_features: NDArray,
        target: NDArray,
        input_feature_labels: list[str],
    ) -> PreparedDataset:
        """Validate arrays and return training data with a newly assigned UUID.

        Array shapes and feature labels follow the TrainingData contract. Its
        TypeError and ValueError exceptions propagate to the caller, and no ID
        is generated for invalid data. The returned object must be passed to
        subsequent training code directly; its ID alone cannot retrieve data.
        """
        training_data = TrainingData(input_features, target, input_feature_labels)
        return PreparedDataset(dataset_id=uuid4(), training_data=training_data)
