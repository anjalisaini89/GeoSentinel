from geosentinel.geospatial.ingestion import validate_raster_file


def test_ingestion_module_imports():
    assert callable(validate_raster_file)