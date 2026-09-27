from pathlib import Path

import rasterio

from geosentinel.geospatial.tiling import tile_raster


def test_tile_raster():
    output_dir = Path("data/processed/test_tiles")

    tiles = tile_raster(
        "data/samples/sample.tif",
        output_dir,
        tile_size=128,
    )

    assert len(tiles) == 4

    for tile in tiles:
        assert tile.exists()

        with rasterio.open(tile) as src:
            assert src.width == 128
            assert src.height == 128
            assert src.count == 3
            assert str(src.crs) == "EPSG:4326"
            assert src.res == (0.0001, 0.0001)


def test_tiles_have_different_spatial_positions():
    output_dir = Path("data/processed/test_tiles")

    tiles = tile_raster(
        "data/samples/sample.tif",
        output_dir,
        tile_size=128,
    )

    bounds = []

    for tile in tiles:
        with rasterio.open(tile) as src:
            bounds.append(src.bounds)

    assert len(set(bounds)) == 4