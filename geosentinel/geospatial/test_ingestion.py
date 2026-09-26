from pathlib import Path

import pytest

from geosentinel.geospatial.ingestion import validate_raster_file


def test_ingestion_module_imports():
    assert callable(validate_raster_file)


def test_missing_raster():
    with pytest.raises(FileNotFoundError):
        validate_raster_file("data/samples/does_not_exist.tif")


def test_unsupported_format(tmp_path: Path):
    test_file = tmp_path / "image.jpg"
    test_file.write_text("not a raster")

    with pytest.raises(ValueError, match="Unsupported raster format"):
        validate_raster_file(test_file)


def test_valid_sample_raster():
    result = validate_raster_file("data/samples/sample.tif")

    assert result["valid"] is True
    assert result["width"] == 256
    assert result["height"] == 256
    assert result["bands"] == 3
    assert result["crs"] == "EPSG:4326"
    assert result["driver"] == "GTiff"