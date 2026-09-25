from pathlib import Path
from typing import Any

import rasterio


def inspect_raster(file_path: str | Path) -> dict[str, Any]:
    """
    Inspect a raster file and return its core geospatial metadata.

    Parameters
    ----------
    file_path:
        Path to a GeoTIFF/COG raster.

    Returns
    -------
    dict
        Raster metadata including dimensions, CRS, resolution,
        bounds, data type, transform and nodata value.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Raster file not found: {file_path}")

    with rasterio.open(file_path) as src:
        metadata = {
            "filename": file_path.name,
            "path": str(file_path),
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

    return metadata