"""Input validation and labelled array behavior for TrainingData."""

import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from app.core.data import TrainingData


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.features = np.arange(6, dtype=float).reshape(3, 2)
        self.target = np.arange(12, dtype=float).reshape(3, 4)
        self.labels = ["temperature", "pressure"]

    def test_valid_inputs_are_stored_as_labelled_dataarrays(self):
        data = TrainingData(self.features, self.target, self.labels)

        self.assertIsInstance(data.input_features, xr.DataArray)
        self.assertIsInstance(data.target, xr.DataArray)
        self.assertEqual(data.input_features.dims, ("sample", "feature"))
        self.assertEqual(data.target.dims, ("sample", "bin"))
        np.testing.assert_array_equal(data.input_features.values, self.features)
        np.testing.assert_array_equal(data.target.values, self.target)
        np.testing.assert_array_equal(data.input_features.coords["feature"], self.labels)
        np.testing.assert_array_equal(data.input_features.coords["sample"], [0, 1, 2])
        np.testing.assert_array_equal(data.target.coords["sample"], [0, 1, 2])
        np.testing.assert_array_equal(data.target.coords["bin"], [0, 1, 2, 3])
        np.testing.assert_array_equal(
            data.input_features.sel(feature="pressure").values, self.features[:, 1]
        )

    def test_mismatched_sample_counts_are_rejected(self):
        for sample_count in (0, 2, 4):
            with self.subTest(target_samples=sample_count):
                with self.assertRaisesRegex(ValueError, "same number of samples"):
                    TrainingData(
                        self.features, np.zeros((sample_count, 4)), self.labels
                    )

    def test_too_few_or_too_many_feature_labels_are_rejected(self):
        for labels in ([], ["temperature"], ["temperature", "pressure", "extra"]):
            with self.subTest(labels=labels):
                with self.assertRaisesRegex(ValueError, "length must match"):
                    TrainingData(self.features, self.target, labels)

    def test_non_2d_arrays_are_rejected(self):
        for name in ("input_features", "target"):
            for array in (np.array(1), np.zeros(3), np.zeros((3, 2, 1))):
                with self.subTest(argument=name, shape=array.shape):
                    args = dict(
                        input_features=self.features,
                        target=self.target,
                        input_feature_labels=self.labels,
                    )
                    args[name] = array
                    with self.assertRaisesRegex(ValueError, f"{name} must be 2D"):
                        TrainingData(**args)

    def test_non_numpy_arrays_are_rejected(self):
        for name in ("input_features", "target"):
            for value in (None, [[1, 2]], xr.DataArray([[1, 2]])):
                with self.subTest(argument=name, value_type=type(value)):
                    args = dict(
                        input_features=self.features,
                        target=self.target,
                        input_feature_labels=self.labels,
                    )
                    args[name] = value
                    with self.assertRaisesRegex(TypeError, f"{name} must be a NumPy array"):
                        TrainingData(**args)

    def test_labels_must_be_a_list_of_strings(self):
        for labels in (
            None,
            "temperature",
            ("temperature", "pressure"),
            np.array(self.labels),
            ["temperature", 1],
            ["temperature", None],
        ):
            with self.subTest(labels=labels):
                with self.assertRaisesRegex(TypeError, "must be a list of strings"):
                    TrainingData(self.features, self.target, labels)

    def test_single_sample_and_single_feature_keep_two_dimensions(self):
        data = TrainingData(np.array([[1.0]]), np.array([[2.0]]), ["temperature"])

        self.assertEqual(data.input_features.shape, (1, 1))
        self.assertEqual(data.target.shape, (1, 1))
        self.assertEqual(data.input_features.sel(feature="temperature").item(), 1.0)

    def test_invalid_counts_are_rejected_before_xarray_conversion(self):
        for target, labels, message in (
            (np.zeros((2, 4)), self.labels, "same number of samples"),
            (self.target, ["temperature"], "length must match"),
        ):
            with self.subTest(message=message):
                with patch("app.core.data.xr.DataArray") as data_array:
                    with self.assertRaisesRegex(ValueError, message):
                        TrainingData(self.features, target, labels)
                    data_array.assert_not_called()

    def test_matching_empty_dimensions_are_preserved(self):
        for samples, parameters, bins in ((0, 2, 4), (3, 0, 4), (3, 2, 0)):
            with self.subTest(samples=samples, parameters=parameters, bins=bins):
                labels = self.labels[:parameters]
                data = TrainingData(
                    np.empty((samples, parameters)), np.empty((samples, bins)), labels
                )
                self.assertEqual(data.input_features.shape, (samples, parameters))
                self.assertEqual(data.target.shape, (samples, bins))
                self.assertEqual(data.input_features.dims, ("sample", "feature"))
                self.assertEqual(data.target.dims, ("sample", "bin"))
                np.testing.assert_array_equal(data.input_features.coords["feature"], labels)

    def test_duplicate_feature_labels_are_preserved(self):
        labels = ["temperature", "temperature"]
        data = TrainingData(self.features, self.target, labels)

        np.testing.assert_array_equal(data.input_features.coords["feature"], labels)
        np.testing.assert_array_equal(data.input_features.values, self.features)


if __name__ == "__main__":
    unittest.main()
