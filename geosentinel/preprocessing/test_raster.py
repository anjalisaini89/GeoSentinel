from pathlib import Path

import numpy as np
import pytest

from geosentinel.preprocessing.raster import (
    load_raster_array,
    normalize_uint8,
    preprocess_raster,
)


def test_load_raster_array():
    array = load_raster_array(
        "data/samples/sample.tif",
        expected_bands=3,
    )

    assert isinstance(array, np.ndarray)
    assert array.shape == (3, 256, 256)
    assert array.dtype == np.uint8


def test_normalize_uint8():
    array = np.array(
        [[[0, 127, 255]]],
        dtype=np.uint8,
    )

    normalized = normalize_uint8(array)

    assert normalized.dtype == np.float32
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0


def test_preprocess_raster():
    array = preprocess_raster(
        "data/samples/sample.tif",
        expected_bands=3,
    )

    assert isinstance(array, np.ndarray)
    assert array.shape == (3, 256, 256)
    assert array.dtype == np.float32

    assert array.min() >= 0.0
    assert array.max() <= 1.0


def test_missing_raster():
    with pytest.raises(FileNotFoundError):
        load_raster_array(
            "data/samples/does_not_exist.tif"
        )


def test_wrong_band_count():
    with pytest.raises(ValueError, match="Expected 4 bands"):
        load_raster_array(
            "data/samples/sample.tif",
            expected_bands=4,
        )


def test_normalize_rejects_non_uint8():
    array = np.array(
        [[[0.0, 1.0]]],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="Expected uint8"):
        normalize_uint8(array)