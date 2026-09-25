from pathlib import Path
from typing import Any

import rasterio


SUPPORTED_EXTENSIONS = {".tif", ".tiff"}


def validate_raster_file(file_path: str | Path) -> dict[str, Any]:
    """
    Validate a raster before it enters the GeoSentinel pipeline.

    Checks:
    - File exists
    - Supported extension
    - Raster can be opened by Rasterio
    - Raster has valid dimensions
    - Raster has at least one band
    - CRS is present
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Raster file not found: {file_path}"
        )

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported raster format: {file_path.suffix}. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    with rasterio.open(file_path) as src:
        if src.width <= 0 or src.height <= 0:
            raise ValueError("Raster has invalid dimensions.")

        if src.count <= 0:
            raise ValueError("Raster contains no bands.")

        if src.crs is None:
            raise ValueError("Raster does not contain a CRS.")

        return {
            "valid": True,
            "filename": file_path.name,
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "crs": str(src.crs),
            "driver": src.driver,
        }