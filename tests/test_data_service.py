"""Dataset preparation and framework independence tests."""

import subprocess
import sys
import unittest
from unittest.mock import patch
from uuid import UUID

import numpy as np

from app.core.data import TrainingData
from app.services.data_service import DataService, PreparedDataset


class DataServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = DataService()
        self.features = np.arange(6, dtype=float).reshape(3, 2)
        self.target = np.arange(12, dtype=float).reshape(3, 4)
        self.labels = ["temperature", "pressure"]

    def test_preparation_returns_uuid_and_training_data(self):
        prepared = self.service.prepare_upload(
            self.features, self.target, self.labels
        )

        self.assertIsInstance(prepared, PreparedDataset)
        self.assertIsInstance(prepared.dataset_id, UUID)
        self.assertEqual(prepared.dataset_id.version, 4)
        self.assertIsInstance(prepared.training_data, TrainingData)
        self.assertEqual(prepared.training_data.input_features.dims, ("sample", "feature"))
        self.assertEqual(prepared.training_data.target.dims, ("sample", "bin"))
        np.testing.assert_array_equal(
            prepared.training_data.input_features.values, self.features
        )
        np.testing.assert_array_equal(prepared.training_data.target.values, self.target)
        np.testing.assert_array_equal(
            prepared.training_data.input_features.coords["feature"], self.labels
        )
        np.testing.assert_array_equal(
            prepared.training_data.input_features.coords["sample"], [0, 1, 2]
        )
        np.testing.assert_array_equal(
            prepared.training_data.target.coords["sample"], [0, 1, 2]
        )
        np.testing.assert_array_equal(
            prepared.training_data.target.coords["bin"], [0, 1, 2, 3]
        )

    def test_separate_preparations_receive_different_ids(self):
        first = self.service.prepare_upload(self.features, self.target, self.labels)
        second = self.service.prepare_upload(self.features, self.target, self.labels)

        self.assertNotEqual(first.dataset_id, second.dataset_id)
        self.assertIsNot(first.training_data, second.training_data)

    def test_invalid_data_propagates_errors_without_assigning_ids(self):
        cases = (
            (self.features, np.zeros((2, 4)), self.labels, ValueError, "same number of samples"),
            (self.features, self.target, ["temperature"], ValueError, "length must match"),
            (self.features, self.target, self.labels + ["extra"], ValueError, "length must match"),
            ([[1, 2]], self.target, self.labels, TypeError, "input_features must be a NumPy array"),
            (self.features, None, self.labels, TypeError, "target must be a NumPy array"),
            (self.features, self.target, ["temperature", 1], TypeError, "list of strings"),
            (np.zeros(3), self.target, self.labels, ValueError, "input_features must be 2D"),
            (self.features, np.zeros(3), self.labels, ValueError, "target must be 2D"),
        )
        for features, target, labels, error, message in cases:
            with self.subTest(message=message, labels=labels):
                with patch("app.services.data_service.uuid4") as generate_id:
                    with self.assertRaisesRegex(error, message):
                        self.service.prepare_upload(features, target, labels)
                    generate_id.assert_not_called()

    def test_conversion_failure_propagates_without_assigning_an_id(self):
        error = ValueError("Dataset conversion failed")
        with patch("app.core.data.xr.DataArray", side_effect=error):
            with patch("app.services.data_service.uuid4") as generate_id:
                with self.assertRaises(ValueError) as caught:
                    self.service.prepare_upload(self.features, self.target, self.labels)
                self.assertIs(caught.exception, error)
                generate_id.assert_not_called()

    def test_service_can_be_imported_and_used_without_fastapi(self):
        code = """
import sys
sys.modules['fastapi'] = None
from app.services.data_service import DataService
import numpy as np
prepared = DataService().prepare_upload(np.ones((2, 1)), np.ones((2, 3)), ['x'])
assert prepared.training_data.target.shape == (2, 3)
assert sys.modules['fastapi'] is None
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
