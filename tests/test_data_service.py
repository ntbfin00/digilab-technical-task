"""Dataset preparation and framework independence tests."""

import subprocess
from io import BytesIO
import lzma
import warnings
import zipfile
import sys
import unittest
from unittest.mock import patch
from uuid import UUID

import numpy as np

from app.core.data import TrainingData
from app.services.data_service import (DataService, PreparedDataset, InvalidArchiveError,
                                       DatasetTooLargeError)


def malformed_npz_archives():
    """Real malformed Unicode and LZMA payloads, shared with HTTP regressions."""
    arrays = dict(input_features=np.ones((3, 2)), target=np.arange(3).reshape(3, 1),
                  input_feature_labels=np.array(["a", "b"]))
    unicode_archive = BytesIO()
    np.savez(unicode_archive, **(arrays | {
        "input_feature_labels": np.array([0x110000, ord("b")], dtype=np.uint32).view("U1"),
    }))
    compressed = BytesIO()
    with zipfile.ZipFile(compressed, "w", compression=zipfile.ZIP_LZMA) as archive:
        for name, array in arrays.items():
            member = BytesIO()
            np.save(member, array, allow_pickle=False)
            archive.writestr(f"{name}.npy", member.getvalue())
        entry = archive.infolist()[0]
        start = entry.header_offset + 30 + len(entry.filename.encode()) + len(entry.extra)
    corrupted = bytearray(compressed.getvalue())
    # Preserve the ZIP LZMA prefix; corrupt its five-byte filter properties.
    corrupted[start + 4:start + 9] = b"\xff" * 5
    return {"unicode": unicode_archive.getvalue(), "lzma": bytes(corrupted)}


class DataServiceTests(unittest.TestCase):
    def archive(self, **overrides):
        values = dict(input_features=np.ones((3, 2)), target=np.ones((3, 1)),
                      input_feature_labels=np.array(["a", "b"]))
        values.update(overrides)
        output = BytesIO()
        np.savez_compressed(output, **values)
        output.seek(0)
        return output

    def test_npz_delegates_once_and_preserves_labels(self):
        with patch.object(self.service, "prepare_upload", wraps=self.service.prepare_upload) as prepare:
            result = self.service.prepare_npz(self.archive(target=self.target))
        prepare.assert_called_once()
        self.assertEqual(prepare.call_args.args[2], ["a", "b"])
        self.assertIsInstance(result.dataset_id, UUID)
        np.testing.assert_array_equal(prepare.call_args.args[1], self.target)
        np.testing.assert_array_equal(result.training_data.target.values, self.target)

    def test_npz_malformed_unicode_and_lzma_are_normalized(self):
        for kind, payload in malformed_npz_archives().items():
            with self.subTest(kind=kind):
                # Verify the fixture actually reproduces the low-level failure.
                with self.assertRaises(SystemError if kind == "unicode" else lzma.LZMAError):
                    with np.load(BytesIO(payload), allow_pickle=False) as archive:
                        if kind == "unicode":
                            archive["input_feature_labels"].tolist()
                        else:
                            archive["input_features"]
                with patch.object(self.service, "prepare_upload") as prepare:
                    with self.assertRaises(InvalidArchiveError):
                        self.service.prepare_npz(BytesIO(payload))
                    prepare.assert_not_called()

    def test_npz_invalid_members_dtypes_and_structures(self):
        cases = [BytesIO(b"not zip"), self.archive(extra=np.ones(1)),
                 self.archive(input_feature_labels=np.array([b"a", b"b"])),
                 self.archive(input_feature_labels=np.array([["a", "b"]])),
                 self.archive(input_features=np.array([[object()]], dtype=object)),
                 self.archive(input_features=np.ones(3)),
                 self.archive(input_features=np.ones((3, 2), dtype=complex)),
                 self.archive(target=np.ones((2, 1)))]
        missing = BytesIO()
        np.savez(missing, target=np.ones((3, 1)))
        cases.append(missing)
        duplicate = self.archive()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(duplicate, "a") as archive:
                archive.writestr("target.npy", b"bad")
        cases.append(duplicate)
        for source in cases:
            with self.subTest(source=source):
                with self.assertRaises(InvalidArchiveError):
                    self.service.prepare_npz(source)

    def test_npz_limits_precede_array_loading(self):
        for limits in ({"max_compressed_bytes": 1}, {"max_expanded_bytes": 1}):
            with self.subTest(limits=limits), patch("app.services.data_service.np.load") as load:
                with self.assertRaises(DatasetTooLargeError):
                    DataService(**limits).prepare_npz(self.archive())
                load.assert_not_called()

    def test_npz_forged_header_rejected_before_allocation(self):
        for shape, error in (((10**12, 2), DatasetTooLargeError), ((3, 2), InvalidArchiveError)):
            header = BytesIO()
            np.lib.format.write_array_header_1_0(header, {
                "descr": "<f8", "fortran_order": False, "shape": shape})
            source = self.archive()
            output = BytesIO()
            with zipfile.ZipFile(source) as original, zipfile.ZipFile(output, "w") as archive:
                for entry in original.infolist():
                    archive.writestr(entry.filename, header.getvalue() if entry.filename == "input_features.npy"
                                     else original.read(entry))
            with patch("app.services.data_service.np.load") as load:
                with self.assertRaises(error):
                    self.service.prepare_npz(output)
                load.assert_not_called()

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
