"""Prepare labelled training datasets independently of any API framework."""

from dataclasses import dataclass
import lzma
import math
from typing import BinaryIO
from uuid import UUID, uuid4
import zipfile
import zlib

import numpy as np
from numpy.typing import NDArray

from app.core.data import TrainingData

MAX_COMPRESSED_BYTES = 32 * 1024 * 1024
MAX_EXPANDED_BYTES = 128 * 1024 * 1024


class InvalidArchiveError(ValueError):
    """An upload is not a supported, well-formed NPZ dataset."""


class DatasetTooLargeError(ValueError):
    """Compressed or expanded dataset exceeds its configured budget."""


@dataclass
class PreparedDataset:
    """A dataset identifier and training data ready to pass to training code."""

    dataset_id: UUID
    training_data: TrainingData


class DataService:
    """Prepare datasets without storing them or retaining service-level state."""

    def __init__(self, *, max_compressed_bytes: int = MAX_COMPRESSED_BYTES,
                 max_expanded_bytes: int = MAX_EXPANDED_BYTES) -> None:
        if max_compressed_bytes <= 0 or max_expanded_bytes <= 0:
            raise ValueError("Dataset limits must be positive")
        self.max_compressed_bytes = max_compressed_bytes
        self.max_expanded_bytes = max_expanded_bytes

    def prepare_npz(self, source: BinaryIO) -> PreparedDataset:
        """Preflight every ZIP entry and NPY header before allocating any arrays.

        The binary stream must be seekable. Never extract entries to disk or
        permit pickle; header sizes must agree with the ZIP directory sizes.
        """
        source.seek(0, 2)
        if source.tell() > self.max_compressed_bytes:
            raise DatasetTooLargeError("Compressed upload exceeds limit")
        source.seek(0)
        names = {"input_features.npy", "target.npy", "input_feature_labels.npy"}
        try:
            with zipfile.ZipFile(source) as archive:
                entries = archive.infolist()
                if len(entries) != 3 or {e.filename for e in entries} != names:
                    raise InvalidArchiveError("Require exactly input_features, target and input_feature_labels")
                if sum(e.file_size for e in entries) > self.max_expanded_bytes:
                    raise DatasetTooLargeError("Expanded archive exceeds limit")
                declared_total = 0
                for entry in entries:
                    with archive.open(entry) as member:
                        version = np.lib.format.read_magic(member)
                        if version == (1, 0):
                            shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                        elif version == (2, 0):
                            shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                        else:
                            raise InvalidArchiveError("Unsupported NPY version")
                        if dtype.itemsize == 0 or any(dimension <= 0 for dimension in shape):
                            raise InvalidArchiveError("Array dimensions and item sizes must be positive")
                        size = math.prod(shape) * dtype.itemsize
                        declared_total += size
                        if declared_total > self.max_expanded_bytes:
                            raise DatasetTooLargeError("Declared array data exceeds limit")
                        if member.tell() + size != entry.file_size:
                            raise InvalidArchiveError("NPY size disagrees with ZIP member size")
                        labels = entry.filename == "input_feature_labels.npy"
                        if dtype.hasobject or (labels and (dtype.kind != "U" or len(shape) != 1)):
                            raise InvalidArchiveError("Labels must be a one-dimensional Unicode array; pickle is forbidden")
                        if not labels and (dtype.kind not in "iuf" or len(shape) != 2):
                            raise InvalidArchiveError("Features and target must be two-dimensional real arrays")
                arrays = {}
                for entry in entries:
                    with archive.open(entry) as member:
                        arrays[entry.filename] = np.load(member, allow_pickle=False)
                try:
                    labels = arrays["input_feature_labels.npy"].tolist()
                except SystemError as exc:
                    raise InvalidArchiveError("Invalid Unicode feature labels") from exc
                return self.prepare_upload(arrays["input_features.npy"], arrays["target.npy"], labels)
        except DatasetTooLargeError:
            raise
        except (ValueError, TypeError, OSError, EOFError, zipfile.BadZipFile,
                NotImplementedError, RuntimeError, zlib.error, lzma.LZMAError) as exc:
            raise InvalidArchiveError("Invalid NPZ dataset") from exc

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
