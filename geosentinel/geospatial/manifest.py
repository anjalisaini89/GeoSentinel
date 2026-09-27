from pathlib import Path
from typing import Any
import json

import rasterio


def build_tile_manifest(
    tiles_dir: str | Path,
) -> list[dict[str, Any]]:
    """
    Build a manifest containing metadata for every GeoTIFF tile.
    """

    tiles_dir = Path(tiles_dir)

    if not tiles_dir.exists():
        raise FileNotFoundError(
            f"Tile directory not found: {tiles_dir}"
        )

    tile_files = sorted(
        list(tiles_dir.glob("*.tif"))
        + list(tiles_dir.glob("*.tiff"))
    )

    manifest: list[dict[str, Any]] = []

    for tile_path in tile_files:
        with rasterio.open(tile_path) as src:
            manifest.append(
                {
                    "tile_id": tile_path.stem,
                    "source": tile_path.name.split("_r")[0] + ".tif",
                    "path": str(tile_path),
                    "width": src.width,
                    "height": src.height,
                    "bands": src.count,
                    "crs": str(src.crs) if src.crs else None,
                    "resolution": list(src.res),
                    "bounds": {
                        "left": src.bounds.left,
                        "bottom": src.bounds.bottom,
                        "right": src.bounds.right,
                        "top": src.bounds.top,
                    },
                    "dtype": list(src.dtypes),
                    "transform": list(src.transform),
                    "nodata": src.nodata,
                    "driver": src.driver,
                }
            )

    return manifest


def save_manifest(
    manifest: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """
    Save a tile manifest to a JSON file.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    return output_path