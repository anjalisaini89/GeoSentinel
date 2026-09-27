import json
from pathlib import Path

from geosentinel.geospatial.manifest import (
    build_tile_manifest,
    save_manifest,
)


def test_build_tile_manifest():
    tiles_dir = Path("data/processed/test_tiles")

    manifest = build_tile_manifest(tiles_dir)

    assert len(manifest) == 4

    for item in manifest:
        assert "tile_id" in item
        assert "source" in item
        assert "path" in item
        assert "width" in item
        assert "height" in item
        assert "bands" in item
        assert "crs" in item
        assert "resolution" in item
        assert "bounds" in item

        assert item["width"] == 128
        assert item["height"] == 128
        assert item["bands"] == 3
        assert item["crs"] == "EPSG:4326"


def test_manifest_tile_paths_exist():
    tiles_dir = Path("data/processed/test_tiles")

    manifest = build_tile_manifest(tiles_dir)

    for item in manifest:
        assert Path(item["path"]).exists()


def test_save_manifest(tmp_path: Path):
    tiles_dir = Path("data/processed/test_tiles")

    manifest = build_tile_manifest(tiles_dir)

    output_path = tmp_path / "manifest.json"

    saved_path = save_manifest(
        manifest,
        output_path,
    )

    assert saved_path.exists()

    with saved_path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)

    assert len(loaded) == 4
    assert loaded[0]["crs"] == "EPSG:4326"
    assert loaded[0]["width"] == 128
    assert loaded[0]["height"] == 128