from pathlib import Path
from typing import Any

import rasterio


def build_tile_manifest(
    tiles_dir: str | Path,
) -> list[dict[str, Any]]:
    """
    Build a manifest containing metadata for every GeoTIFF tile.

    Parameters
    ----------
    tiles_dir:
        Directory containing generated GeoTIFF tiles.

    Returns
    -------
    list[dict[str, Any]]
        Metadata for each tile.
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
                    "resolution": src.res,
                    "bounds": {
                        "left": src.bounds.left,
                        "bottom": src.bounds.bottom,
                        "right": src.bounds.right,
                        "top": src.bounds.top,
                    },
                    "dtype": src.dtypes,
                    "transform": tuple(src.transform),
                    "nodata": src.nodata,
                    "driver": src.driver,
                }
            )

    return manifest