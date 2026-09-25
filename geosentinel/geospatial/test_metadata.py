from geosentinel.geospatial.metadata import inspect_raster


def test_metadata_module_imports():
    assert callable(inspect_raster)