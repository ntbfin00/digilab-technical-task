"""Validated, labelled input and target arrays for machine learning training."""

import numpy as np
import xarray as xr
from numpy.typing import NDArray


class TrainingData:
    """Store NumPy training arrays as labelled xarray DataArrays.

    Args:
        input_features: Array shaped (N_samples, N_parameters).
        target: Array shaped (N_samples, N_bins), sharing the input sample count.
        input_feature_labels: One string label per input parameter.

    Raises:
        TypeError: Arrays are not NumPy arrays or labels are not a list of strings.
        ValueError: Arrays are not 2D, sample counts differ, or label count differs
            from the number of input parameters.

    Feature labels are available as ``input_features.coords["feature"]``.
    Data values are not copied; mutations of the original arrays may be reflected
    in the stored DataArrays. Validation applies at construction time.
    """

    input_features: xr.DataArray
    target: xr.DataArray

    def __init__(
        self,
        input_features: NDArray,
        target: NDArray,
        input_feature_labels: list[str],
    ) -> None:
        for name, array in (("input_features", input_features), ("target", target)):
            if not isinstance(array, np.ndarray):
                raise TypeError(f"{name} must be a NumPy array")
            if array.ndim != 2:
                raise ValueError(
                    f"{name} must be 2D; received shape {array.shape}"
                )

        if not isinstance(input_feature_labels, list) or not all(
            isinstance(label, str) for label in input_feature_labels
        ):
            raise TypeError("input_feature_labels must be a list of strings")

        if input_features.shape[0] != target.shape[0]:
            raise ValueError(
                "input_features and target must have the same number of samples; "
                f"received {input_features.shape[0]} and {target.shape[0]}"
            )
        if len(input_feature_labels) != input_features.shape[1]:
            raise ValueError(
                "input_feature_labels length must match the number of input "
                f"parameters; received {len(input_feature_labels)} labels for "
                f"{input_features.shape[1]} parameters"
            )

        samples = np.arange(input_features.shape[0])
        self.input_features = xr.DataArray(
            input_features,
            dims=("sample", "feature"),
            coords={"sample": samples, "feature": input_feature_labels},
            name="input_features",
        )
        self.target = xr.DataArray(
            target,
            dims=("sample", "bin"),
            coords={"sample": samples, "bin": np.arange(target.shape[1])},
            name="target",
        )
