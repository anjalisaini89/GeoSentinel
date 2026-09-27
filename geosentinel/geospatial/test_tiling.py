from pathlib import Path

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